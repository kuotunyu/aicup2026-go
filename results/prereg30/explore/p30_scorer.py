"""prereg30 exploration (competition 2): the v2 scorer (PLAN.md 1.1).

V2Scorer is results/prereg08/p1/p1_lib.ProbeScorer with the SAME forward passes and the SAME P1 values (the parent's
position_features runs unchanged; the model is wrapped only to keep a reference to the outputs it already returns).
From those outputs it also keeps, per position:
    lmax  max of the masked log-softmax of the main policy head, channel 0 (log of the largest prior)
    lead  main head miscvalue[:, 2] * lead_multiplier (KataGo scoreLead, side to move, points)
    winp  softmax(main head value logits)[:, 0] (win probability of the side to move)
No extra forward pass, no search, no extra metadata profile.

Importing p1_lib checks the pinned hs_lib and installs its data/local guard. No file is written by this module.
"""

import json
import os
import sys

sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
P1_DIR = os.path.join(REPO, "results", "prereg08", "p1")
for _p in (HERE, P1_DIR, os.path.join(REPO, "tools")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import p1_lib as P  # noqa: E402  (pin check, data/local guard)
import p30_lib as V  # noqa: E402

H = P.H
P30Error = V.P30Error


# ================================================================ scorer
class _Capture:
    """Callable stand-in for the model: forwards the call unchanged and keeps the returned outputs."""

    def __init__(self, model, owner):
        self._model = model
        self._owner = owner

    def __call__(self, *a, **k):
        out = self._model(*a, **k)
        self._owner._last_out = out
        return out

    def __getattr__(self, name):
        return getattr(self._model, name)


class V2Scorer(P.ProbeScorer):
    """p1_lib.ProbeScorer plus the per-position lmax / lead / winp of PLAN.md 1.1 from the same forward passes.

    position_features(feats) returns exactly what ProbeScorer returns (the parent method runs unchanged);
    afterwards self.last_extra is float32 [T, 3] = (lmax, lead, winp)."""

    def __init__(self, model, *a, **k):
        super().__init__(model, *a, **k)
        self.lead_multiplier = float(model.lead_multiplier)
        self.model = _Capture(self.model, self)
        self._last_out = None
        self._heads = []
        self.last_extra = None

    def _forward(self, b, g, m):
        self._last_out = None
        pol, trunk = super()._forward(b, g, m)
        out = self._last_out
        if out is None:
            raise V.P30Error("model outputs were not captured")
        # main head: out[0] = (policy, value, miscvalue, ...); keep the padded batch as the parent evaluated it
        self._heads.append((pol, out[0][1].float(), out[0][2][:, 2].float()))
        return pol, trunk

    def position_features(self, feats):
        torch = self.torch
        self._heads = []
        res = super().position_features(feats)
        T = len(feats["move"])
        extra = np.zeros((T, 3), dtype=np.float32)
        per = self.batch
        starts = list(range(0, T, per))
        if len(starts) != len(self._heads):
            raise V.P30Error(f"{len(self._heads)} forward passes captured, {len(starts)} expected")
        with torch.inference_mode():
            for (pol, val, lead_raw), s in zip(self._heads, starts):
                e = min(T, s + per)
                Pn = e - s
                # the same rows (with the same padding) as the parent, so lmax comes from the same batch shape
                rows = np.arange(s, e) if not self.pad or Pn == per else \
                    np.concatenate([np.arange(s, e), np.full(per - Pn, e - 1)])
                lm = torch.as_tensor(feats["legal"][rows], device=self.device)
                lp = torch.log_softmax(pol.masked_fill(~lm, float("-inf")), dim=-1)
                lmax = lp.max(dim=1).values
                winp = torch.softmax(val, dim=1)[:, 0]
                lead = lead_raw * self.lead_multiplier
                extra[s:e] = torch.stack([lmax, lead, winp], dim=1)[:Pn].cpu().numpy()
        self._heads = []
        if not np.isfinite(extra).all():
            raise V.P30Error("non-finite lmax / lead / winp")
        self.last_extra = extra
        return res


def code_record():
    rec = dict(P.code_shas())
    rec.update(V.own_code_shas())
    return json.dumps(rec, sort_keys=True)

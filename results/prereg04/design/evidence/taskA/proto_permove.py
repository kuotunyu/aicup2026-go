"""Prototype (CPU): derive per-move samples of the target player from the official 500-plane rank
features, and check them against the SGF move list. Fit games only (c10data.load_rows part='fit'),
a deterministic sample. No data/local file is read. No GPU."""
import json, os, re, sys, time, hashlib
import numpy as np
import torch
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "tools"))
sys.path.insert(0, os.path.join(REPO, "third_party", "aicup2026_tutorial"))
from c10data import load_rows
from utils import SGFParseRankPrediction
torch.set_num_threads(4)
H = 8
MOVE = re.compile(r";([BW])\[([a-s]{2})\]")

def per_move_samples(x, is_black, H=H):
    """x: uint8 [B,500,19,19] (official: plane 2t black, 2t+1 white, oldest state first);
    is_black: bool [B] target plays black. Returns dict of per-move tensors for the target's moves."""
    B = x.shape[0]
    v = x.view(B, 250, 2, 361)
    v = torch.where(is_black[:, None, None, None], v, v.flip(2))      # [:, :, 0] target, [:, :, 1] opponent
    prev, cur = v[:, :-1], v[:, 1:]                                    # 249 transitions
    gain = cur > prev
    own = gain[:, :, 0].sum(-1)
    opp = gain[:, :, 1].sum(-1)
    first_nonempty = v[:, 0].reshape(B, -1).any(-1)                    # window starts mid-game (truncated)
    t = torch.arange(249)
    hist_ok = (~first_nonempty[:, None]) | (t[None, :] >= H - 1)       # full H-state history inside the window
    valid = (own == 1) & (opp == 0) & hist_ok
    label = gain[:, :, 0].to(torch.uint8).argmax(-1)                  # [B,249]
    b, ti = valid.nonzero(as_tuple=True)                              # position before the move = state ti
    pad = torch.cat([torch.zeros(B, H - 1, 2, 361, dtype=v.dtype), v], 1)
    hidx = ti[:, None] + (H - 1) - torch.arange(H)[None, :]           # states ti, ti-1, ..., ti-H+1
    planes = pad[b[:, None], hidx]                                     # [P,H,2,361]
    P = planes.shape[0]
    inp = torch.cat([planes[:, :, 0], planes[:, :, 1],                # H own planes, H opponent planes
                     is_black[b].to(torch.uint8)[:, None, None].expand(P, 1, 361),
                     torch.ones(P, 1, 361, dtype=torch.uint8)], 1).view(P, 2 * H + 2, 19, 19)
    lab = label[b, ti]
    legal = (planes[:, 0, 0] == 0) & (planes[:, 0, 1] == 0)           # empty points before the move
    return {"inp": inp, "label": lab.long(), "game": b, "t": ti, "legal": legal,
            "n_invalid_own_gain": int(((own > 1) | ((own == 1) & (opp > 0))).sum())}

rows = load_rows(os.path.join(REPO, "data", "training"), os.path.join(REPO, "results", "split_v1", "players_split.csv"), part="fit", local_dir=None)
rows = [r for r in rows if int(hashlib.sha256(("p04proto|" + r["game_id"] + r["rank"]).encode()).hexdigest()[:8], 16) % 400 == 0]
print("sample games", len(rows), flush=True)
parser = SGFParseRankPrediction()
t0 = time.time()
feats = []
for r in rows:
    f, e = parser.features_from_sgf_content(r["sgf_content"])
    assert e is None
    feats.append(f)
t_parse = time.time() - t0
x = torch.from_numpy(np.stack(feats))
isb = torch.tensor([r["color"] == "B" for r in rows])
t0 = time.time()
out = per_move_samples(x, isb)
t_derive = time.time() - t0
# reference from SGF: target moves in the window (moves N-248..N), history rule, coordinates
ref_n, ref_labels, bad = 0, [], 0
exp_by_game = []
for gi, r in enumerate(rows):
    mv = MOVE.findall(r["sgf_content"])
    n = len(mv)
    lo = max(0, n - 249)
    trunc = n + 1 > 250
    lab = []
    for k in range(lo, n):               # move k+1 (1-indexed); transition index within the window
        c, p = mv[k]
        ti = k - lo                        # state before the move is window state ti when truncated
        if not trunc:
            ti = (249 - n) + k             # short game: padding (249-n) states, root at 249-n
        if c != r["color"]:
            continue
        if trunc and ti < H - 1:
            continue
        col = ord(p[0]) - 97; row_from_top = ord(p[1]) - 97; row = 18 - row_from_top
        lab.append((ti, col * 19 + row))  # official board_np[y=col, x=row]
    exp_by_game.append(lab)
    ref_n += len(lab)
got = {}
for g, ti, l in zip(out["game"].tolist(), out["t"].tolist(), out["label"].tolist()):
    got.setdefault(g, []).append((ti, l))
mismatch = sum(1 for gi in range(len(rows)) if got.get(gi, []) != exp_by_game[gi])
legal_ok = bool(out["legal"][torch.arange(len(out["label"])), out["label"]].all())
res = {"games": len(rows), "positions": int(len(out["label"])), "reference_positions": ref_n,
       "games_mismatch": mismatch, "label_always_empty_point": legal_ok,
       "n_invalid_own_gain": out["n_invalid_own_gain"],
       "input_shape": list(out["inp"].shape), "parse_ms_per_game_1core": 1e3 * t_parse / len(rows),
       "derive_ms_per_game_cpu4": 1e3 * t_derive / len(rows),
       "mean_positions_per_game": len(out["label"]) / len(rows)}
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "proto_permove.json"), "w"), indent=1)

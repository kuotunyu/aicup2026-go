"""prereg06 explore / katago: multiply-accumulate count per position for two KataGo configs (random weights, CPU).

Used only to turn public speed reports for b28c512nbt into a rough b18c384nbt-humanv0 speed (REPORT.md section 4).
No weights are downloaded. Writes results/prereg06/explore/katago/kg_macs.json.

Usage (repo root, Git Bash):
  .venv/Scripts/python.exe results/prereg06/explore/katago/kg_macs.py --katago-python <clone>/python
"""

import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from kg_guard import install_guard  # noqa: E402

install_guard()
import torch  # noqa: E402


def count(model_cls, config, with_meta):
    model = model_cls(config, 19)
    model.initialize()
    model.eval()
    macs = [0]

    def conv_hook(mod, inp, out):
        macs[0] += out.numel() // out.shape[0] * (mod.in_channels // mod.groups) * mod.kernel_size[0] * mod.kernel_size[1]

    def lin_hook(mod, inp, out):
        macs[0] += out.numel() // out.shape[0] * mod.in_features

    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            m.register_forward_hook(conv_hook)
        elif isinstance(m, torch.nn.Linear):
            m.register_forward_hook(lin_hook)
    x = torch.zeros(1, 22, 19, 19)
    x[:, 0] = 1.0
    g = torch.zeros(1, 19)
    meta = torch.zeros(1, 192) if with_meta else None
    with torch.no_grad():
        model(x, g, input_meta=meta)
    return {"params": int(sum(p.numel() for p in model.parameters())), "macs": int(macs[0]),
            "gflops": 2 * macs[0] / 1e9}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--katago-python", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "kg_macs.json"))
    args = ap.parse_args()
    torch.set_num_threads(4)
    sys.path.insert(0, os.path.abspath(args.katago_python))
    from katago.train import modelconfigs
    from katago.train.model_pytorch import Model

    res = {}
    c18 = copy.deepcopy(modelconfigs.config_of_name["b18c384nbt-mish-meta"])
    c18["version"] = 14
    res["b18c384nbt-mish-meta(v14)"] = count(Model, c18, True)
    res["b18c384nbt"] = count(Model, copy.deepcopy(modelconfigs.config_of_name["b18c384nbt"]), False)
    res["b28c512nbt"] = count(Model, copy.deepcopy(modelconfigs.config_of_name["b28c512nbt"]), False)
    res["ratio_b28_over_b18meta"] = res["b28c512nbt"]["macs"] / res["b18c384nbt-mish-meta(v14)"]["macs"]
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

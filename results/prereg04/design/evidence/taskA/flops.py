"""Exact forward FLOPs (2 x MACs; conv + linear) of the prereg03 model and candidate per-move models. CPU."""
import json, os, sys, time
import torch, torch.nn as nn, torch.nn.functional as F
REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "third_party", "aicup2026_tutorial"))
from network import GoRankResNet, ResidualBlock
torch.set_num_threads(4)

class PerMoveNet(nn.Module):
    def __init__(self, cin=18, ch=128, blocks=10, levels=10, head_ch=16):
        super().__init__()
        self.conv_in = nn.Conv2d(cin, ch, 3, padding=1); self.bn_in = nn.BatchNorm2d(ch)
        self.layers = nn.Sequential(*[ResidualBlock(ch) for _ in range(blocks)])
        self.head1 = nn.Conv2d(ch, levels * head_ch, 3, padding=1); self.bn_h = nn.BatchNorm2d(levels * head_ch)
        self.head2 = nn.Conv2d(levels * head_ch, levels, 1, groups=levels)
    def forward(self, x):
        o = F.relu(self.bn_in(self.conv_in(x)))
        o = self.layers(o)
        o = F.relu(self.bn_h(self.head1(o)))
        return self.head2(o).flatten(2)          # [P, 10, 361] logits per level

def flops(model, shape):
    total = [0]
    def conv_hook(m, i, o):
        total[0] += 2 * o.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    def lin_hook(m, i, o):
        total[0] += 2 * o.numel() * m.in_features
    hs = [m.register_forward_hook(conv_hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    hs += [m.register_forward_hook(lin_hook) for m in model.modules() if isinstance(m, nn.Linear)]
    with torch.no_grad():
        model.eval()(torch.zeros(shape))
    for h in hs: h.remove()
    return total[0]

out = {}
m3 = GoRankResNet(in_channels=501)
out["prereg03_501x256x20"] = {"fwd_gflops": flops(m3, (1, 501, 19, 19)) / 1e9,
                              "params_m": sum(p.numel() for p in m3.parameters()) / 1e6}
for name, (b, c) in {"6x96": (6, 96), "8x128": (8, 128), "10x128": (10, 128), "12x128": (12, 128), "10x160": (10, 160), "6x64": (6, 64)}.items():
    m = PerMoveNet(ch=c, blocks=b)
    out[name] = {"fwd_gflops": flops(m, (1, 18, 19, 19)) / 1e9, "params_m": sum(p.numel() for p in m.parameters()) / 1e6}
# measured: prereg03 arm A bf16 training 60.3 ms/step at batch 256 (results/prereg03/rank_a/timings.json)
ms, bs = 60.3, 256
train_flops_p03 = 3 * out["prereg03_501x256x20"]["fwd_gflops"] * 1e9
eff = train_flops_p03 * bs / (ms / 1e3)
out["calibration"] = {"prereg03_train_step_ms": ms, "batch": bs, "effective_train_tflops": eff / 1e12}
# prereg03 final dumps: inner 73,556 + val 19,156 games x (8 syms + 1 swap) forwards in 160.3 s (fp32/TF32, incl. val parsing)
fw = (73556 + 19156) * 9 * out["prereg03_501x256x20"]["fwd_gflops"] * 1e9
out["calibration"]["effective_dump_tflops_upper_bound_incl_parse"] = fw / 160.34 / 1e12
for name in ("6x96", "8x128", "10x128", "12x128", "10x160", "6x64"):
    f = out[name]["fwd_gflops"] * 1e9
    out[name]["train_pos_per_s_at_p03_efficiency"] = eff / (3 * f)
    out[name]["train_pos_per_s_at_50pct_of_p03_eff"] = 0.5 * eff / (3 * f)
    out[name]["fp32_dump_pos_per_s_at_p03_dump_eff"] = (fw / 160.34) / f
# CPU sanity: forward of 10x128 on 256 positions
m = PerMoveNet().eval(); x = torch.zeros(256, 18, 19, 19)
with torch.no_grad():
    m(x); t0 = time.time(); m(x); out["cpu_fwd_ms_256pos_10x128_4threads"] = (time.time() - t0) * 1e3
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "flops.json"), "w"), indent=1)

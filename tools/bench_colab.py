"""Single-core CPU timing of the costly steps in the two prereg-01 notebooks,
plus exact per-sample FLOPs of the two models. CPU only; no GPU, no training."""
import csv, gzip, io, json, os, sys, time, random
import numpy as np
import torch

REPO = r"D:\AI-Portfolio\competitions\aicup2026-go"
sys.path.insert(0, os.path.join(REPO, "third_party", "aicup2026_tutorial"))
from utils import SGFParseRankPrediction, SGFParsePlayerIdentification  # noqa: E402
from network import GoRankResNet, GoPlayerResNet  # noqa: E402

torch.set_num_threads(1)
csv.field_size_limit(10**8)
LEVELS = ["D", "C", "B", "A", "1D", "2D", "3D", "4D", "5D", "6D"]
rows = []
for lv in LEVELS:
    with open(os.path.join(REPO, "data", "training", f"train_{lv}.csv"), newline="", encoding="utf-8") as f:
        r = csv.reader(f); next(r)
        for i, rec in enumerate(r):
            if i % 3001 == 0:
                rows.append(rec)
            if len(rows) >= (LEVELS.index(lv) + 1) * 30:
                break
print("games sampled", len(rows), "mean moves",
      sum(rec[4].count(";") - 1 for rec in rows) / len(rows))
out = {"games": len(rows)}

# 1. rank features: parse + gzip (webdataset .tar.gz uses gzip; level 9 and 6 both timed)
rp = SGFParseRankPrediction()
t0 = time.perf_counter(); feats = []
for rec in rows:
    f, e = rp.features_from_sgf_content(rec[4]); feats.append(f)
t_parse = (time.perf_counter() - t0) / len(rows)
buf = io.BytesIO(); np.save(buf, feats[0]); raw = buf.getvalue()
t0 = time.perf_counter()
for f in feats[:60]:
    b = io.BytesIO(); np.save(b, f); gzip.compress(b.getvalue(), compresslevel=9)
t_gz9 = (time.perf_counter() - t0) / 60
t0 = time.perf_counter()
for f in feats[:60]:
    b = io.BytesIO(); np.save(b, f); gzip.compress(b.getvalue(), compresslevel=6)
t_gz6 = (time.perf_counter() - t0) / 60
comp = gzip.compress(raw, 9)
t0 = time.perf_counter()
for _ in range(200):
    a = np.load(io.BytesIO(gzip.decompress(comp))); torch.from_numpy(a).float()
t_rank_load = (time.perf_counter() - t0) / 200
out["rank"] = {"parse_ms_per_game": t_parse * 1e3, "gzip9_ms": t_gz9 * 1e3, "gzip6_ms": t_gz6 * 1e3,
               "raw_kb": len(raw) / 1024, "gz_kb": len(comp) / 1024,
               "loader_decode_ms_per_sample": t_rank_load * 1e3}
del feats

# 2. player features: parse (inference mode) + gzip level 6 (as in the notebook)
pp = SGFParsePlayerIdentification()
t0 = time.perf_counter(); pfe = []
for rec in rows:
    w = pp.extract_features(rec[4], rec[3].lower(), mode="inference"); pfe.append(np.asarray(w, dtype=np.uint8))
t_pparse = (time.perf_counter() - t0) / len(rows)
t0 = time.perf_counter(); gz = []
for a in pfe:
    b = io.BytesIO()
    with gzip.GzipFile(fileobj=b, mode="wb", compresslevel=6) as g:
        np.save(g, a)
    gz.append(b.getvalue())
t_pgz = (time.perf_counter() - t0) / len(pfe)
# 3. player training: load one whole game file and pick one window (x3 per triplet)
t0 = time.perf_counter(); n = 0
for _ in range(2):
    for g in gz:
        with gzip.GzipFile(fileobj=io.BytesIO(g), mode="rb") as f:
            arr = np.load(f)
        _ = arr[random.randrange(len(arr))]; n += 1
t_pload = (time.perf_counter() - t0) / n
out["player"] = {"parse_ms_per_game": t_pparse * 1e3, "gzip6_ms_per_game": t_pgz * 1e3,
                 "raw_kb_per_game": sum(a.nbytes for a in pfe) / len(pfe) / 1024,
                 "gz_kb_per_game": sum(len(g) for g in gz) / len(gz) / 1024,
                 "windows_per_game": sum(len(a) for a in pfe) / len(pfe),
                 "load_one_window_ms": t_pload * 1e3,
                 "cpu_ms_per_training_step_batch100": t_pload * 1e3 * 300}

# 4. exact forward FLOPs (multiply-adds x2) via hooks
def flops(model, shape):
    total = [0]
    def conv_hook(m, i, o):
        k = m.kernel_size[0] * m.kernel_size[1]
        total[0] += 2 * o.numel() * (m.in_channels // m.groups) * k
    def lin_hook(m, i, o):
        total[0] += 2 * o.numel() * m.in_features
    hs = []
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d): hs.append(m.register_forward_hook(conv_hook))
        if isinstance(m, torch.nn.Linear): hs.append(m.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(torch.zeros(shape))
    for hk in hs: hk.remove()
    return total[0]
rm, pm = GoRankResNet(), GoPlayerResNet()
out["rank"]["fwd_gflops_per_sample"] = flops(rm.eval(), (1, 500, 19, 19)) / 1e9
out["rank"]["params_m"] = sum(p.numel() for p in rm.parameters()) / 1e6
out["player"]["fwd_gflops_per_window"] = flops(pm.eval(), (1, 17, 19, 19)) / 1e9
out["player"]["params_m"] = sum(p.numel() for p in pm.parameters()) / 1e6
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(os.path.dirname(__file__), "bench_out.json"), "w"), indent=1)

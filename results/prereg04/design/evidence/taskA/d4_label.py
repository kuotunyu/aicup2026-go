"""CPU check: with D4_PERM[s] = d4(arange(361), s) and x' = x.gather(perm), the label moves to argmax(perm == label)."""
import torch
def d4(x, s):
    if s == 0:
        return x
    y = torch.flip(x, dims=(3,)) if s >= 4 else x
    return torch.rot90(y, k=s % 4, dims=(2, 3)).contiguous()
PERM = torch.stack([d4(torch.arange(361).view(1, 1, 19, 19), s).reshape(361) for s in range(8)])
INV = torch.argsort(PERM, dim=1)          # INV[s][q] = p with PERM[s][p] = q
ok = True
for s in range(8):
    for lab in range(361):
        onehot = torch.zeros(1, 1, 19, 19); onehot.view(-1)[lab] = 1
        new = int(d4(onehot, s).reshape(-1).argmax())
        ok &= (new == int(INV[s][lab]))
print("label' = INV[s][label] for all 8 x 361:", ok)

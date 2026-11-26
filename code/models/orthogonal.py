# 正交投影基底生成器。
# 与 code/tools/_archive/t_collapse_diag.py 的 seed 42 生成序逐位一致。
import torch


def random_orthogonal_matrices(seed=42):
    g = torch.Generator().manual_seed(seed)
    dims = {'x3': 128, 'x4': 256, 'x5': 512}
    out = {}
    for lv, n in dims.items():
        for p in ('proj_ir', 'proj_tmpl'):
            a = torch.randn(n, n, generator=g)
            q, _ = torch.linalg.qr(a)
            out[f'{lv}:{p}'] = q
    return out

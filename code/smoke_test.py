# 端到端冒烟测试：模型前反向 + 数据集读取 + 评测
# 用法: python smoke_test.py
import os
import torch

from models.localizer import Localizer
from losses.guided_focal import GuidedFocalMSELoss


def test_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    B, N, H, W = 2, 3, 384, 640
    ir = torch.randn(B, 3, H, W).to(device)
    tmpl = torch.randn(B, N, 3, 224, 224).to(device)
    heatmap = torch.rand(B, 1, H, W).to(device)
    supp = torch.zeros(B, 1, H, W).to(device)
    crit = GuidedFocalMSELoss()

    for mod in ('correlation',):
        model = Localizer(encoder='resnet18', modulator=mod, pretrained=False).to(device)
        out = model(ir, tmpl)
        assert out.shape == (B, 1, H, W), f"{mod} 输出形状错: {out.shape}"
        loss = crit(out, heatmap, supp)
        loss.backward()
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"[{mod:12s}] out={tuple(out.shape)} loss={loss.item():.4f} "
              f"params={n_params:.2f}M  N={N} 多模板 OK")
    print("模型冒烟测试通过\n")


def test_dataset():
    from data.cstloc_dataset import CSTLocDataset
    root = os.environ.get('CSTLOC_DATA_ROOT', '/home/user/Code/Datasets/CSTLoc')
    for mode in ('train', 'test'):
        ds = CSTLocDataset(root=root, split=mode, mode=mode,
                           n_templates=1)
        print(f"[dataset/{mode}] {len(ds)} 样本")
        if len(ds):
            s = ds[0]
            shapes = {k: (tuple(v.shape) if torch.is_tensor(v) else v)
                      for k, v in s.items()}
            print(f"   sample keys: {shapes}")
    print("数据集冒烟测试通过")


if __name__ == '__main__':
    test_models()
    test_dataset()

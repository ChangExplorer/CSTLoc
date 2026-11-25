# CSTLoc-UAV 数据集：RGB 模板 → IR 场景图 定位
#
# 目录：<root>/<split>/<instance>/{rgb,ir}/{template,search}/
#   ir/search/*.json  labelme：shapes[0]=目标(target:true)，其余=干扰物(target:false)
#   rgb/template/*.jpg 模板切片（可见光参考）
#
# 配对（简版）：对每个 IR search 帧，从 RGB 模板中随机取满足 Δt≥1s 的帧。
# 训练 N=1；评测可指定 N（均匀取多个模板）。
# GT 坐标系为 IR（定位发生在 IR 图）。
import json
import random
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .heatmap import generate_gaussian_heatmap

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def _to_tensor(img_rgb):
    """HWC uint8 RGB → CHW float 归一化 tensor。"""
    t = torch.from_numpy(img_rgb).float().permute(2, 0, 1) / 255.0
    mean = torch.tensor(NORM_MEAN).view(3, 1, 1)
    std = torch.tensor(NORM_STD).view(3, 1, 1)
    return (t - mean) / std


def _letterbox(img, out_hw):
    """保长宽比 resize + pad（居中，黑边）。返回 (图, scale, pad_left, pad_top)。"""
    H, W = out_hw
    h, w = img.shape[:2]
    scale = min(W / w, H / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((H, W, 3), dtype=img.dtype)
    pad_l, pad_t = (W - nw) // 2, (H - nh) // 2
    canvas[pad_t:pad_t + nh, pad_l:pad_l + nw] = resized
    return canvas, scale, pad_l, pad_t


class CSTLocDataset(Dataset):
    def __init__(self, root, split='test', mode='train',
                 img_size=(384, 640), patch_size=224,
                 sigma_factor=0.3, n_templates=1, min_gap_frames=30,
                 instances=None):
        """
        Args:
            root:          CSTLoc-UAV 根目录
            split:         'train' | 'test'
            mode:          'train'（返回热图/抑制掩码）| 'test'（返回 GT 中心/尺寸/元信息）
            n_templates:   每个样本的模板数量 N
            min_gap_frames: 配对最小帧间隔（30 帧 @30fps = 1s）
            instances:     限定实例名列表（None=全部）
        """
        assert mode in ('train', 'test')
        self.mode = mode
        self.img_size = img_size
        self.patch_size = patch_size
        self.sigma_factor = sigma_factor
        self.n_templates = n_templates
        self.min_gap = min_gap_frames

        split_dir = Path(root) / split
        self.samples = []   # (inst_dir, search_json_path, [template_frame_ids])
        for inst_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            if instances is not None and inst_dir.name not in instances:
                continue
            # RGB→IR 方向：跳过 RGB 不可用的夜间实例（meta.json 的 rgb_usable=false）
            meta_path = inst_dir / 'meta.json'
            if meta_path.exists():
                _meta = json.loads(meta_path.read_text(encoding='utf-8'))
                if _meta.get('rgb_usable') is False:
                    continue
            tmpl_dir = inst_dir / 'rgb' / 'template'
            tmpl_frames = sorted(int(p.stem) for p in tmpl_dir.glob('*.jpg'))
            if not tmpl_frames:
                continue
            for sj in sorted((inst_dir / 'ir' / 'search').glob('*.json')):
                sf = int(sj.stem)
                cands = [t for t in tmpl_frames if abs(sf - t) >= self.min_gap]
                if len(cands) >= n_templates:
                    self.samples.append((inst_dir, sj, cands))

    def __len__(self):
        return len(self.samples)

    def _pick_templates(self, cands):
        """训练随机取 N；评测取时间轴均匀分位的 N（确定性、嵌套可比）。"""
        if self.n_templates == 1:
            return [random.choice(cands)] if self.mode == 'train' else [cands[len(cands) // 2]]
        if self.mode == 'train':
            return random.sample(cands, self.n_templates)
        n = self.n_templates
        idx = [round((2 * k + 1) * (len(cands) - 1) / (2 * n)) for k in range(n)]
        return [cands[i] for i in sorted(set(idx))]

    def _load_template(self, inst_dir, frame_id):
        p = inst_dir / 'rgb' / 'template' / f'{frame_id:06d}.jpg'
        img = cv2.imread(str(p))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.patch_size, self.patch_size),
                         interpolation=cv2.INTER_LINEAR)
        return _to_tensor(img)

    def __getitem__(self, idx):
        inst_dir, sj, cands = self.samples[idx]
        meta = json.loads(sj.read_text(encoding='utf-8'))

        # IR search 图
        ir_path = sj.with_suffix('.jpg')
        ir = cv2.cvtColor(cv2.imread(str(ir_path)), cv2.COLOR_BGR2RGB)
        ir, scale, pad_l, pad_t = _letterbox(ir, self.img_size)

        def map_box(pts):
            (x1, y1), (x2, y2) = pts
            return [x1 * scale + pad_l, y1 * scale + pad_t,
                    x2 * scale + pad_l, y2 * scale + pad_t]

        shapes = meta['shapes']
        tgt = map_box(shapes[0]['points'])
        distractors = [map_box(s['points']) for s in shapes[1:]
                       if not s['flags'].get('target')]

        # 模板
        tf = self._pick_templates(cands)
        templates = torch.stack([self._load_template(inst_dir, f) for f in tf])  # (N,3,224,224)

        ir_tensor = _to_tensor(ir)

        if self.mode == 'test':
            cx = (tgt[0] + tgt[2]) / 2
            cy = (tgt[1] + tgt[3]) / 2
            bw, bh = tgt[2] - tgt[0], tgt[3] - tgt[1]
            return {
                'ir': ir_tensor,
                'templates': templates,
                'gt_center': torch.tensor([cx, cy]),
                'obj_size': torch.tensor([bw, bh]),
                'difficulty': meta['flags'].get('difficulty', ''),
                'num_distractors': meta['flags'].get('num_distractors') or 0,
                'instance': inst_dir.name,
            }

        # train：目标高斯热图 + 干扰物抑制掩码
        heatmap = generate_gaussian_heatmap(
            np.array([tgt], dtype=np.float32), self.img_size, self.sigma_factor)
        supp = generate_gaussian_heatmap(
            np.array(distractors, dtype=np.float32), self.img_size, self.sigma_factor
        ) if distractors else np.zeros(self.img_size, dtype=np.float32)

        return {
            'ir': ir_tensor,
            'templates': templates,
            'heatmap': torch.from_numpy(heatmap).unsqueeze(0),
            'suppression': torch.from_numpy(supp).unsqueeze(0),
        }

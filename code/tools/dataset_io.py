"""Dataset access: frames, templates, protocol enumeration, donors.

All functions operate on the dataset rooted at CSTLOC_DATA_ROOT with layout
<split>/<instance>/{rgb/template, ir/search}. Box coordinates returned by
load_frame are in the letterboxed (384x640) frame.
"""
import cv2
import json
import torch

from data.cstloc_dataset import _letterbox, _to_tensor
from tools.image_ops import IMG_SIZE, norm224
from tools.paths import data_root

ROOT = data_root()


def map_box(pts, scale, pad_l, pad_t):
    (x1, y1), (x2, y2) = pts
    return [x1 * scale + pad_l, y1 * scale + pad_t,
            x2 * scale + pad_l, y2 * scale + pad_t]


def load_frame(inst_dir, frame_id):
    """读 IR search 帧（letterbox 后）+ GT/干扰物框（letterbox 坐标系）。"""
    sj = inst_dir / 'ir' / 'search' / f'{frame_id:06d}.json'
    meta = json.loads(sj.read_text(encoding='utf-8'))
    ir_path = sj.with_suffix('.jpg')
    ir = cv2.cvtColor(cv2.imread(str(ir_path)), cv2.COLOR_BGR2RGB)
    ir, scale, pad_l, pad_t = _letterbox(ir, IMG_SIZE)
    shapes = meta['shapes']
    tgt = map_box(shapes[0]['points'], scale, pad_l, pad_t)
    distractors = [map_box(s['points'], scale, pad_l, pad_t)
                   for s in shapes[1:] if not s['flags'].get('target')]
    cx = (tgt[0] + tgt[2]) / 2
    cy = (tgt[1] + tgt[3]) / 2
    bw, bh = tgt[2] - tgt[0], tgt[3] - tgt[1]
    return {
        'ir': _to_tensor(ir),
        'gt_center': torch.tensor([cx, cy]),
        'obj_size': torch.tensor([bw, bh]),
        'gt_box': tgt,
        'distractor_boxes': distractors,
        'num_distractors_actual': len(distractors),
        'difficulty': meta['flags'].get('difficulty', ''),
        'num_distractors_flag': meta['flags'].get('num_distractors') or 0,
    }


def letterboxed_bgr(split, inst, f):
    """Letterboxed IR search frame as BGR uint8 (crop input for candidates)."""
    rgb, _, _, _ = _letterbox(cv2.cvtColor(cv2.imread(str(
        ROOT / split / inst / 'ir/search' / f'{f:06d}.jpg')), cv2.COLOR_BGR2RGB), IMG_SIZE)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def template_ids(split, inst):
    """协议模板枚举：rgb/template 下全部 jpg 帧号（升序）。"""
    return sorted(int(p.stem) for p in (ROOT / split / inst / 'rgb/template').glob('*.jpg'))


def frame_ids(split, inst):
    return sorted(int(p.stem) for p in (ROOT / split / inst / 'ir/search').glob('*.json'))


def protocol_frame_trials(split, inst, f, tmpls=None):
    """N协议单帧trial列表：|f−t|≥30；val/test只调用同一extract_peaks_and_crops。"""
    tmpls = template_ids(split, inst) if tmpls is None else tmpls
    return [{'split': split, 'inst': inst, 't': int(t), 'f': int(f)}
            for t in tmpls if abs(f - t) >= 30]


def list_instances(split):
    """实例全集：跳过 rgb_usable=false；返回 [(inst, template_ids, frame_ids)]。"""
    out = []
    for inst in sorted(p.name for p in (ROOT / split).iterdir() if p.is_dir()):
        mp = ROOT / split / inst / 'meta.json'
        if mp.exists() and json.loads(mp.read_text(encoding='utf-8')).get('rgb_usable') is False:
            continue
        out.append((inst, template_ids(split, inst), frame_ids(split, inst)))
    return out


def load_template_tensor(split, inst, t):
    im = cv2.cvtColor(cv2.imread(str(ROOT / split / inst / 'rgb/template' / f'{t:06d}.jpg')),
                      cv2.COLOR_BGR2RGB)
    assert im is not None
    return norm224(cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR))


def train_donor_for(inst):
    """train目标：同类train实例字典序循环下一实例，保证非目标。"""
    prefix = inst.rsplit('_', 1)[0] + '_'
    peers = sorted(p.name for p in (ROOT / 'train').iterdir()
                   if p.is_dir() and p.name.startswith(prefix) and template_ids('train', p.name))
    i = peers.index(inst)
    donor = peers[(i + 1) % len(peers)]
    assert donor != inst
    ts = template_ids('train', donor)
    return donor, {t: min(ts, key=lambda u: (abs(u - t), u)) for t in template_ids('train', inst)}

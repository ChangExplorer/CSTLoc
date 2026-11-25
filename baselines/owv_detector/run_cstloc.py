# baselines/owv_detector/run.py —— 开放词汇检测器臂（GroundingDINO，任务书 2.4 臂 B）
#
# 每（实例, 场景帧）以实例类别名为文本提示做一次 GroundingDINO 前向，
# 取置信度最高框的中心为预测（原图 1920×1080 坐标）；每模板重复写该预测
# （检测器不使用模板，这是它与实例级定位器的本质差别）。
# 产物：results.jsonl + peaks.jsonl（top-32 检测中心+分数）。
# 断点：owv_progress.json（实例级）。
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'references' / 'repos' / 'GroundingDINO'))
import cv2  # noqa: E402
import torch  # noqa: E402

from groundingdino.models import build_model  # noqa: E402
from groundingdino.util.slconfig import SLConfig  # noqa: E402
from groundingdino.util.utils import clean_state_dict  # noqa: E402

GD = ROOT / 'references' / 'repos' / 'GroundingDINO'
DATA = Path(os.environ.get('CSTLOC_DATA_ROOT', '/home/user/Code/Datasets/CSTLoc-UAV_v2'))
MIN_GAP = 30


def load_model():
    cfg = SLConfig.fromfile(str(GD / 'groundingdino/config/GroundingDINO_SwinT_OGC.py'))
    cfg.device = 'cuda'
    model = build_model(cfg)
    ckpt = torch.load(ROOT / 'references' / 'weights' / 'groundingdino_swint_ogc.pth',
                      map_location='cpu')
    model.load_state_dict(clean_state_dict(ckpt['model']), strict=False)
    model = model.cuda().eval()
    return model


@torch.no_grad()
def detect(model, img_bgr, caption):
    """一次前向 → [(cx, cy, score), ...]（原图坐标）。"""
    from groundingdino.datasets import transforms as T
    caption = caption.lower().strip()
    if not caption.endswith('.'):
        caption += '.'
    image_pil, image = None, None
    from PIL import Image
    image_pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    transform = T.Compose([T.RandomResize([800], max_size=1333),
                           T.ToTensor(), T.Normalize([0.485, 0.456, 0.406],
                                                   [0.229, 0.224, 0.225])])
    image, _ = transform(image_pil, None)
    image = image.cuda()
    outputs = model(image[None], captions=[caption])
    logits = outputs['pred_logits'].sigmoid()[0].cpu()   # (nq, 256)
    boxes = outputs['pred_boxes'][0].cpu()               # (nq, 4), cxcywh 归一化
    scores = logits.max(dim=1)[0]
    H, W = img_bgr.shape[:2]
    res = []
    for b, s in zip(boxes, scores):
        cx, cy = float(b[0]) * W, float(b[1]) * H
        res.append((cx, cy, float(s)))
    res.sort(key=lambda x: -x[2])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='test', choices=['test'])
    ap.add_argument('--instances', default=None, help='逗号分隔子集（冒烟用）')
    args = ap.parse_args()
    torch.manual_seed(0)
    model = load_model()
    here = Path(os.environ.get('OWV_OUT_DIR', str(Path(__file__).resolve().parent)))
    prog_f = here / 'owv_progress.json'
    done = json.loads(prog_f.read_text()) if prog_f.exists() else {}
    res_f = open(here / 'results.jsonl', 'a', encoding='utf-8')
    pk_f = open(here / 'peaks.jsonl', 'a', encoding='utf-8')
    t0 = time.time()
    for inst_dir in sorted((DATA / args.split).iterdir()):
        inst = inst_dir.name
        if args.instances and inst not in args.instances.split(','):
            continue
        if done.get(inst):
            continue
        cls = inst.rsplit('_', 1)[0]
        tmpls = sorted(int(p.stem) for p in (inst_dir / 'rgb' / 'template').glob('*.jpg'))
        frames = sorted(int(p.stem) for p in (inst_dir / 'ir' / 'search').glob('*.json'))
        lines = []
        for f in frames:
            img = cv2.imread(str(inst_dir / 'ir' / 'search' / f'{f:06d}.jpg'))
            dets = detect(model, img, cls)
            if dets:
                cx, cy, sc = dets[0]
            else:
                cx, cy, sc = -1000.0, -1000.0, 0.0   # 未检出哨兵（必定 miss）
            for t in tmpls:
                if abs(f - t) < MIN_GAP:
                    continue
                lines.append(json.dumps({'inst': inst, 'tmpl_mod': 'rgb', 't': t,
                                         'scene_mod': 'ir', 'f': f,
                                         'pred': [cx, cy]}, ensure_ascii=False))
            pk_f.write(json.dumps({'inst': inst, 't': tmpls[0] if tmpls else -1, 'f': f,
                                   'peaks': [[round(x, 2), round(y, 2), round(s, 6)]
                                             for x, y, s in dets[:32]]},
                                  ensure_ascii=False) + '\n')
        res_f.write('\n'.join(lines) + ('\n' if lines else ''))
        res_f.flush()
        done[inst] = 1
        prog_f.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'[{inst}] {len(lines)} trials（累计 {time.time() - t0:.0f}s）', flush=True)
    res_f.close()
    pk_f.close()


if __name__ == '__main__':
    main()

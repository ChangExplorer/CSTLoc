# 错模板探针：实例判别 vs 类别显著性诊断（任务 2 / 阶段 3 任务 4 回归）
#
# 加载指定 encoder 的两个 checkpoint（corr / film best_model.pth），
# 对 test 实例 car_008 / car_010 / car_015 各取按帧号升序前 10 帧
# 且 difficulty==hard、num_distractors>=3 的 search 帧（不够 10 帧全取，如实记录），
# 三种模板条件各测一遍：
#   a) 本实例正确模板（该实例全部 rgb 模板帧升序取中位）
#   b) 错模板（train 实例 car_048 的 rgb 模板帧升序取中位）
#   c) 随机噪声图（固定种子 torch.manual_seed(0)，确定性）
# 每个条件报 PCK@0.5（0.5·max(w,h) 阈值，与 engine.evaluate 一致）。
#
# 确定性规则：选帧/选模板规则固定，随机噪声固定种子 → 跑几次结果必须一样。
#   （阶段 3 回归要求：选帧规则/donor/三条件与实验 A 严格同口径，帧号列表逐帧相同）
# 输出：code/diag_out/probe_<tag>_<模型>.json（帧号列表 + 逐帧结果，可复核）。
#
# 用法:
#   python tools/wrong_template_probe.py
#       → 实验 A 回归（encoder=resnet18，checkpoints/resnet_{corr,film}，tag=默认）
#   python tools/wrong_template_probe.py --encoder dinov2 --ckpt-prefix dinov2_ --tag dinov2
#       → 阶段 3 回归（encoder=dinov2，checkpoints/dinov2_{corr,film}，tag=dinov2）
# 依赖：现有环境 torch/cv2/numpy（不新增依赖）；import data.cstloc_dataset 的
#       _letterbox/_to_tensor 与 engine.evaluate.peak_center（不改任何现有文件）。
import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

CODE = Path(__file__).resolve().parents[1]          # code/
ROOT = CODE.parent / 'dataset' / 'CSTLoc-UAV'       # dataset/CSTLoc-UAV
sys.path.insert(0, str(CODE))

from data.cstloc_dataset import _letterbox, _to_tensor  # noqa: E402
from engine.evaluate import peak_center                 # noqa: E402
from models.localizer import Localizer                  # noqa: E402

IMG_SIZE = (384, 640)
PATCH = 224
PCK_THRESH = 0.5
MAX_FRAMES = 10
TARGETS = ['car_008', 'car_010', 'car_015']
DONOR = 'car_048'
NOISE_SEED = 0
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUT_DIR = CODE / 'diag_out'


def median_frame(inst_dir):
    """该实例全部 rgb 模板帧升序取中位（确定性）。"""
    frames = sorted(int(p.stem) for p in (inst_dir / 'rgb' / 'template').glob('*.jpg'))
    return frames[len(frames) // 2]


def load_template(inst_dir, frame_id):
    """加载模板并归一化，返回 (3,224,224) tensor（与 data.cstloc_dataset 同流程）。"""
    p = inst_dir / 'rgb' / 'template' / f'{frame_id:06d}.jpg'
    img = cv2.imread(str(p))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (PATCH, PATCH), interpolation=cv2.INTER_LINEAR)
    return _to_tensor(img)


def select_frames(inst_dir):
    """按帧号升序取前 10 帧 hard 且 num_distractors>=3 的 search 帧。"""
    out = []
    for sj in sorted((inst_dir / 'ir' / 'search').glob('*.json'), key=lambda p: int(p.stem)):
        meta = json.loads(sj.read_text(encoding='utf-8'))
        flags = meta.get('flags', {})
        if flags.get('difficulty') == 'hard' and (flags.get('num_distractors') or 0) >= 3:
            out.append(int(sj.stem))
            if len(out) >= MAX_FRAMES:
                break
    return out


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


def main():
    global ROOT
    # 确定性（缺陷修复 2026-09-06）：消除探针 ±1~2 帧的运行间抖动（J3 时代观测记录）。
    # 评测纯前向；K 实验证据链：CUBLAS_WORKSPACE_CONFIG + cudnn.deterministic 后前向位级一致。
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default=str(CODE.parent / 'dataset' / 'CSTLoc-UAV'),
                    help='数据集根目录（默认 v1 路径）')
    ap.add_argument('--frames-json', default=None,
                    help='val 口径帧号 JSON（probe_val_frames.json）；传入时按 val_hard 节运行（实例在 val split）')
    ap.add_argument('--encoder', default='resnet18', choices=['resnet18', 'dinov2'])
    ap.add_argument('--ckpt-prefix', default='resnet_',
                    help='checkpoint 目录前缀，如 resnet_ / dinov2_')
    ap.add_argument('--ckpt-dir', default=None,
                    help='单模型模式：直接指定 checkpoint 目录（corr 调制器），如 c1_pilot')
    ap.add_argument('--identity-expert', default=None,
                    help='J2 注入模型：head_v*.pt 路径；给出时以 use_id 加载（j2_expert 单模型 corr），'
                         '并增加 d_null 条件（正确模板 + id_emb=null 向量）')
    ap.add_argument('--tag', default='',
                    help='输出文件前缀（默认空=实验A命名 probe_<模型>.json）')
    args = ap.parse_args()
    ROOT = Path(args.data_root)
    tag = args.tag + '_' if args.tag else ''
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f'device={DEVICE} | encoder={args.encoder} | targets={TARGETS} | donor={DONOR} | max_frames={MAX_FRAMES}')

    # 模型（默认 corr/film 双调制器；--ckpt-dir 指定时单模型只跑 corr）
    def resolve_dir(name, mod_name):
        cands = [f'{args.ckpt_prefix}{name}', f'{args.ckpt_prefix}{mod_name}']
        for c in cands:
            if (CODE / 'checkpoints' / c).exists():
                return c
        raise FileNotFoundError(f'未找到 checkpoint 目录: {cands}')
    if args.ckpt_dir:
        ckpt_map = [('corr', args.ckpt_dir, 'correlation')]
    else:
        ckpt_map = [('corr', resolve_dir('corr', 'correlation'), 'correlation'),
                    ('film', resolve_dir('film', 'film'), 'film')]
    models = {}
    dnull_models = {}
    for name, ckpt_name, mod in ckpt_map:
        ckpt = CODE / 'checkpoints' / ckpt_name / 'best_model.pth'
        if args.identity_expert:
            m = Localizer(encoder=args.encoder, modulator=mod, pretrained=True,
                          use_id=True, identity_expert=args.identity_expert,
                          mask_null_p=0.0).to(DEVICE)
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=True)
            m.eval()
            # d_null 用：mask_null_p=1.0 → mask_null=True 时全部样本 id_emb=null
            md = Localizer(encoder=args.encoder, modulator=mod, pretrained=True,
                           use_id=True, identity_expert=args.identity_expert,
                           mask_null_p=1.0).to(DEVICE)
            md.load_state_dict(torch.load(ckpt, map_location=DEVICE), strict=True)
            md.eval()
            models[name] = m
            dnull_models[name] = md
        else:
            m = Localizer(encoder=args.encoder, modulator=mod, pretrained=True).to(DEVICE)
            m.load_state_dict(torch.load(ckpt, map_location=DEVICE))
            m.eval()
            models[name] = m
        print(f'模型 {name} 加载完成: {ckpt}')
    conds_all = ('a_own', 'b_wrong', 'c_noise') + (('d_null',) if args.identity_expert else ())

    # 模板（三个条件；b/c 对所有 search 帧共用同一模板，a 逐实例）
    donor_dir = ROOT / 'train' / DONOR
    donor_tmpl = load_template(donor_dir, median_frame(donor_dir)).unsqueeze(0)  # (1,3,224,224)
    donor_frame = median_frame(donor_dir)
    torch.manual_seed(NOISE_SEED)
    noise_tmpl = torch.randn(1, 3, PATCH, PATCH)

    # 目标实例表（先于 doc 构造判定：--frames-json 生效时 meta.targets 须写实际运行的 val 锚点实例）
    if args.frames_json:
        fj = json.load(open(args.frames_json, encoding='utf-8'))
        frame_map = {inst: v['frames'] for inst, v in fj['val_hard']['targets'].items()}
        split_dir = 'val'
        targets_use = list(frame_map.keys())
    else:
        frame_map = None
        split_dir = 'test'
        targets_use = TARGETS

    doc = {
        'meta': {
            'checkpoints': {n: str(CODE / 'checkpoints' / c / 'best_model.pth')
                            for n, c, _ in ckpt_map},
            'encoder': args.encoder,
            'targets': targets_use, 'donor': DONOR, 'donor_template_frame': donor_frame,
            'max_frames': MAX_FRAMES, 'noise_seed': NOISE_SEED, 'pck_thresh': PCK_THRESH,
            'frame_selection_rule': '帧号升序, difficulty==hard, num_distractors>=3, 取前10帧',
            'template_rule': '该实例全部 rgb 模板帧升序取中位; 错模板=car_048 中位帧; 噪声=固定种子 randn',
            'pck_definition': 'dist(pred_center, gt_center) <= 0.5*max(w,h)',
        },
        'instances': {},
        'summary': {'corr': {}, 'film': {}},
    }

    for inst in targets_use:
        inst_dir = ROOT / split_dir / inst
        frames = frame_map[inst] if frame_map is not None else select_frames(inst_dir)
        own_tmpl = load_template(inst_dir, median_frame(inst_dir)).unsqueeze(0)
        own_frame = median_frame(inst_dir)
        cond_tmpls = {'a_own': own_tmpl, 'b_wrong': donor_tmpl, 'c_noise': noise_tmpl}
        if args.identity_expert:
            cond_tmpls['d_null'] = own_tmpl   # 正确模板 + id_emb=null（J2 §5.2）
        print(f'\n[实例 {inst}] 选中帧 {len(frames)} 个: {frames} | 正确模板帧={own_frame}')

        inst_doc = {
            'frames': frames,
            'own_template_frame': own_frame,
            'wrong_template_frame': donor_frame,
            'frames_detail': {},
        }
        # 汇总：{model: {cond: {'hits': n, 'total': n}}}
        agg = {m: {c: {'hits': 0, 'total': 0} for c in cond_tmpls} for m in models}

        for f in frames:
            s = load_frame(inst_dir, f)
            fd = {
                'difficulty': s['difficulty'],
                'num_distractors_flag': s['num_distractors_flag'],
                'num_distractors_actual': s['num_distractors_actual'],
                'gt_center': [float(s['gt_center'][0]), float(s['gt_center'][1])],
                'obj_size': [float(s['obj_size'][0]), float(s['obj_size'][1])],
                'gt_box': s['gt_box'],
                'distractor_boxes': s['distractor_boxes'],
                'models': {},
            }
            for mname, m in models.items():
                fd['models'][mname] = {}
                for cname, tmpl in cond_tmpls.items():
                    m_use = dnull_models[mname] if cname == 'd_null' else m
                    with torch.no_grad():
                        logits = m_use(s['ir'].unsqueeze(0).to(DEVICE),
                                       tmpl.unsqueeze(0).to(DEVICE),
                                       mask_null=(cname == 'd_null'))
                    pred = peak_center(logits)[0].cpu()
                    dist = torch.norm(pred - s['gt_center'])
                    thresh = PCK_THRESH * torch.maximum(s['obj_size'][0], s['obj_size'][1])
                    hit = bool((dist <= thresh).item())
                    fd['models'][mname][cname] = {
                        'hit': hit,
                        'pred': [float(pred[0]), float(pred[1])],
                        'dist_to_gt': float(dist.item()),
                    }
                    agg[mname][cname]['hits'] += int(hit)
                    agg[mname][cname]['total'] += 1
            inst_doc['frames_detail'][str(f)] = fd

        doc['instances'][inst] = inst_doc
        for mname in models:
            doc['summary'][mname][inst] = {}
            for cname in cond_tmpls:
                h = agg[mname][cname]['hits']
                t = agg[mname][cname]['total']
                doc['summary'][mname][inst][cname] = {'pck': round(h / t, 4) if t else None,
                                                      'hits': h, 'total': t}
                print(f"  {mname:5s} {cname:8s} PCK@0.5={h/t:.4f} ({h}/{t})")

    for mname in models:
        for cname in conds_all:
            hs = [doc['summary'][mname][i][cname]['hits'] for i in targets_use]
            ts = [doc['summary'][mname][i][cname]['total'] for i in targets_use]
            doc['summary'][mname]['ALL'] = doc['summary'][mname].get('ALL', {})
            doc['summary'][mname]['ALL'][cname] = {'pck': round(sum(hs) / sum(ts), 4) if sum(ts) else None,
                                                   'hits': sum(hs), 'total': sum(ts)}
        if args.identity_expert:
            a = doc['summary'][mname]['ALL']['a_own']['hits']
            b = doc['summary'][mname]['ALL']['b_wrong']['hits']
            d = doc['summary'][mname]['ALL']['d_null']['hits']
            doc['summary'][mname]['ALL']['a_minus_b'] = a - b
            doc['summary'][mname]['ALL']['a_minus_null'] = a - d

    for mname in models:
        out_path = OUT_DIR / f'probe_{tag}{mname}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(doc['summary'][mname], f, ensure_ascii=False, indent=2)
    # 全量（含逐帧）落盘
    full_path = OUT_DIR / f'probe_{tag}full.json'
    with open(full_path, 'w', encoding='utf-8') as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    print(f'\n落盘: {full_path} + probe_{tag}corr.json / probe_{tag}film.json')


if __name__ == '__main__':
    main()

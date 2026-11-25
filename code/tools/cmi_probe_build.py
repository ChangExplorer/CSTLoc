# 任务书 3.4（CMI 探针）阶段 1：trial 采样 + e116b_s42 top-20 候选 + 特征缓存 + 标签落盘
# 说明：这是历史 CMI 探针的完整入口；发布包中的 03_build_candidates.sh 只复用本文件的
#       build_locator / label_peaks，不需要运行本文件的 main()。
#
# 口径（2026-10-27 预登记，PROGRESS 阶段 64）：
#   采样：np.random.RandomState(42) 全局单流，按 (split: train→val, inst 升序, t 升序)，
#         对 eligible（|f−t|≥30）无放回抽 3 帧；断言 train=13158、val=4065。
#   候选：e116b_s42 定位器（CSTLOC_PROJ_ADAPTER=orth、BATCH_T=12、use_id 路径与 probe_ocf 同式）
#         → sigmoid 热图 → nms_peaks(K=20, min_dist=32)（letterbox 384×640 坐标）。
#   标签：峰心 ∈ GT 框（闭区间）= 2（正）；∈ 任一同类干扰物框 = 1（难负）；否则 = 0（易负）。
#   特征（仓库外 /home/user/Code/cmi_probe_cache/，fp32 torch.save）：
#     候选（按 (f,x,y) 去重）：DINOv2 intermediate_gap(taps 15,23) 2048-d（norm224 预处理，
#     crop_patch 与 V″ 逐位同式）+ CLIP ViT-L-14/openai 768-d 裸向量（open_clip 预处理）。
#     模板：2048-d gap（val 只读复用 n_cache_tmpl.pt；train 按 n 模块 stage1 同式自建：
#     resize224 INTER_LINEAR + norm224 + get_intermediate_layers）+ CLIP 768-d（resize224）。
#   G0 门（phase check）：复刻 n_e116b_s42_trials.jsonl 抽样 trial 的 top-5 峰，坐标须全等、
#     峰值差 <1e-5，否则停机。
#   GPU：硬断言当前设备名含 "4090"（PCI 序=1，由 CUDA_DEVICE_ORDER=PCI_BUS_ID
#     CUDA_VISIBLE_DEVICES=1 暴露为 cuda:0）；n_cache_tmpl.pt sha256 硬断言。
#
# 用法（code/ 目录下）：
#   CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
#     /home/user/anaconda3/envs/viir/bin/python tools/cmi_probe_build.py --phase check|smoke|full|manifest
import argparse
import hashlib
import gc
import json
import os
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))
sys.path.insert(0, str(CODE.parent / 'baselines' / 'clip_rerank' / 'vendor'))
os.environ['CSTLOC_PROJ_ADAPTER'] = 'orth'

import tools.n_single_template_eval as n  # noqa: E402
from data.cstloc_dataset import _letterbox  # noqa: E402
from models.encoders.dinov2_encoder import DINOv2TemplateEncoder  # noqa: E402
from models.localizer import Localizer  # noqa: E402
from tools.v_verifier_rerank import IMG_SIZE, MIN_DIST, crop_patch, nms_peaks, norm224  # noqa: E402
from tools.wrong_template_probe import load_frame  # noqa: E402

K = 20                  # top-20 候选（V″ 同口径）
SEED = 42
BATCH_T = 12            # 与 n 模块一致
MIN_GAP = 30
DEVICE = 'cuda'
ROOT = Path(os.environ.get('CSTLOC_DATA_ROOT', '/home/user/Code/Datasets/CSTLoc'))
EXT = Path('/home/user/Code/cmi_probe_cache')   # 仓库外特征缓存（大 .pt 不入库）
OUT_DIR = CODE / 'diag_out'
CKPT = CODE / 'checkpoints' / 'e116b_s42' / 'best_model.pth'
CLIP_W = Path.home() / '.cache' / 'clip' / 'ViT-L-14.pt'  # openai 官方权重（b23 臂同款，只读）
CACHE = n.OUT_DIR / f'n_cache_tmpl{n.SUF}.pt'
CACHE_SHA256 = '79327010399d583ec48a19952b59513b9d30ab78d0ad58e63412a3f2f3c50334'
TRIALS_REF = n.OUT_DIR / f'n_e116b_s42{n.SUF}_trials.jsonl'

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def assert_env():
    name = torch.cuda.get_device_name(0)
    assert '4090' in name, f'本探针只允许 4090（PCI 序=1），当前 cuda:0 = {name}'
    assert sha256(CACHE) == CACHE_SHA256, f'{CACHE.name} sha256 漂移'
    assert CKPT.exists() and CLIP_W.exists(), '缺 e116b_s42 ckpt 或 CLIP 权重'


def instances(split):
    for d in sorted(p for p in (ROOT / split).iterdir() if p.is_dir()):
        tmpls = sorted(int(p.stem) for p in (d / 'rgb' / 'template').glob('*.jpg'))
        frames = sorted(int(p.stem) for p in (d / 'ir' / 'search').glob('*.json'))
        if tmpls and frames:
            yield d.name, tmpls, frames


def sample_trials():
    """seed 42 全局单流；返回 {split: [ {split,inst,t,f}, ... ]}。断言 13158/4065。"""
    rng = np.random.RandomState(SEED)
    out = {}
    for split in ('train', 'val'):
        recs = []
        for inst, tmpls, frames in instances(split):
            for t in tmpls:
                elig = [f for f in frames if abs(f - t) >= MIN_GAP]
                assert len(elig) >= 3, f'{split}/{inst} t={t} 可抽帧不足 3（{len(elig)}）'
                idx = rng.choice(len(elig), 3, replace=False)
                for f in sorted(elig[int(i)] for i in idx):
                    recs.append({'split': split, 'inst': inst, 't': t, 'f': int(f)})
        out[split] = recs
    assert len(out['train']) == 13158, f'train trials={len(out["train"])}'
    assert len(out['val']) == 4065, f'val trials={len(out["val"])}'
    return out


def build_locator():
    # 默认发布定位器；消融可经 env 指定不同定位器/模块，默认行为逐位不变。
    ckpt = os.environ.get('CSTLOC_LOCATOR_CKPT', str(CKPT))
    modulator = os.environ.get('CSTLOC_LOCATOR_MODULATOR', 'correlation')
    use_id = os.environ.get('CSTLOC_LOCATOR_USE_ID', '1') == '1'
    expert = os.environ.get('CSTLOC_LOCATOR_EXPERT', 'random') if use_id else None
    model = Localizer(encoder='dinov2', modulator=modulator, pretrained=True,
                      ir_encoder='infmae', use_id=use_id, identity_expert=expert,
                      mask_null_p=0.0).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location='cpu'), strict=True)
    model.eval()
    return model


def build_clip():
    """返回 embed(arrs): list[(224,224,3) uint8 RGB] → (N,768) fp32 numpy（裸向量）。"""
    import open_clip
    from torchvision import transforms
    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained=str(CLIP_W), device=DEVICE)
    model.eval()
    mean = std = None
    for tr in preprocess.transforms:
        if isinstance(tr, transforms.Normalize):
            mean, std = tr.mean, tr.std
    assert mean is not None
    mean = torch.tensor(mean).view(1, 3, 1, 1).to(DEVICE)
    std = torch.tensor(std).view(1, 3, 1, 1).to(DEVICE)

    @torch.no_grad()
    def embed(arrs):
        outs = []
        for s in range(0, len(arrs), 64):
            x = torch.from_numpy(np.stack(arrs[s:s + 64])).to(DEVICE)
            x = x.permute(0, 3, 1, 2).float().div_(255.0)
            x = (x - mean) / std
            outs.append(model.encode_image(x, normalize=False).float().cpu())
        return torch.cat(outs).numpy()

    return embed


@torch.no_grad()
def build_tmpl_cache(split, insts, enc, clip_embed):
    """按实例分片 {str(t): {...}}；val 的 maps/gap 只读复用 n_cache_tmpl。返回 LazyTmpl。"""
    val_cache = torch.load(CACHE, map_location='cpu') if split == 'val' else None
    for inst in insts:
        shard = EXT / f'cmi_probe_tmpl_{split}_{inst}.pt'
        if shard.exists():
            continue
        out = {}
        tmpls = sorted(int(p.stem) for p in (ROOT / split / inst / 'rgb' / 'template').glob('*.jpg'))
        imgs = []
        for t in tmpls:
            im = cv2.cvtColor(cv2.imread(str(ROOT / split / inst / 'rgb' / 'template'
                                           / f'{t:06d}.jpg')), cv2.COLOR_BGR2RGB)
            imgs.append(cv2.resize(im, (224, 224), interpolation=cv2.INTER_LINEAR))
        clips = torch.from_numpy(clip_embed(imgs))
        for s0 in range(0, len(tmpls), 32):
            xs = torch.stack([norm224(imgs[i]) for i in range(s0, min(s0 + 32, len(tmpls)))])
            xs = xs.to(DEVICE)
            maps, gaps = None, None
            if split == 'train':
                raw = enc.backbone.get_intermediate_layers(xs, n=[7, 15, 23], reshape=True)
                maps = [m.cpu() for m in raw]  # fp32（与 n_cache_tmpl 一致）
                tok = enc.backbone.get_intermediate_layers(xs, n=[15, 23], reshape=False)
                gaps = torch.cat([tt.mean(dim=1) for tt in tok], dim=1).cpu()
            for j, t in enumerate(tmpls[s0:s0 + 32]):
                d = {}
                if split == 'train':
                    d['m7'], d['m15'], d['m23'] = maps[0][j], maps[1][j], maps[2][j]
                    d['gap'] = gaps[j]
                else:
                    e = val_cache[inst][str(t)]
                    d['m7'], d['m15'], d['m23'], d['gap'] = e['m7'], e['m15'], e['m23'], e['gap']
                d['clip'] = clips[s0 + j]
                out[str(t)] = d
        torch.save(out, shard)
        del out
        gc.collect()
        print(f'  [tmpl {split}] {inst}: {len(tmpls)} 模板', flush=True)
    if val_cache is not None:
        del val_cache
        gc.collect()
    return LazyTmpl(split)


class LazyTmpl:
    """按需加载单实例模板分片，只驻留当前实例（12GB 内存放不下整合并缓存）。"""

    def __init__(self, split):
        self.split = split
        self._cur = None
        self._cache = None

    def __getitem__(self, inst):
        if inst != self._cur:
            self._cache = torch.load(EXT / f'cmi_probe_tmpl_{self.split}_{inst}.pt',
                                     map_location='cpu')
            self._cur = inst
            gc.collect()
        return self._cache


def in_box(px, py, b):
    return b[0] <= px <= b[2] and b[1] <= py <= b[3]


def label_peaks(peaks, gt_box, dboxes):
    labs = []
    for x, y, _v in peaks:
        if in_box(x, y, gt_box):
            labs.append(2)
        elif any(in_box(x, y, b) for b in dboxes):
            labs.append(1)
        else:
            labs.append(0)
    return labs


@torch.no_grad()
def gen_split(split, trials, tcache, suffix=''):
    """跑定位器 → top-20 峰 + 标签 + 候选特征缓存；写 cmi_probe_trials_{split}{suffix}.jsonl。"""
    model = build_locator()
    enc = DINOv2TemplateEncoder(pretrained=True).to(DEVICE).eval()
    clip_embed = build_clip()
    out_path = OUT_DIR / f'cmi_probe_trials_{split}{suffix}.jsonl'
    prog_path = EXT / f'cmi_probe_build_progress_{split}{suffix}.json'
    done = set(json.loads(prog_path.read_text(encoding='utf-8'))) if prog_path.exists() else set()
    if out_path.exists() and not done:
        raise RuntimeError(f'{out_path.name} 已存在但无进度文件，零覆写纪律：先人工核对再删')
    out_f = open(out_path, 'a', encoding='utf-8')
    by_inst = {}
    for tr in trials:
        by_inst.setdefault(tr['inst'], []).append(tr)
    t0 = time.time()
    for inst in sorted(by_inst):
        if inst in done:
            continue
        shard = EXT / f'cmi_probe_cand_{split}_{inst}.pt'
        cand = torch.load(shard, map_location='cpu') if shard.exists() else {}
        by_frame = {}
        for tr in by_inst[inst]:
            by_frame.setdefault(tr['f'], []).append(tr)
        lines = []
        for f, ftrs in sorted(by_frame.items()):
            fr = load_frame(ROOT / split / inst, f)
            ir_x = fr['ir'].unsqueeze(0).to(DEVICE)
            heats = []
            for s in range(0, len(ftrs), BATCH_T):
                feats = [tcache[inst][str(tr['t'])] for tr in ftrs[s:s + BATCH_T]]
                logits = n.forward_cached(model, True, ir_x, feats)
                heats.append(torch.sigmoid(logits)[:, 0].cpu().numpy())
            heats = np.concatenate(heats)
            need = []
            for tr, hn in zip(ftrs, heats):
                tr['peaks'] = nms_peaks(hn, K, MIN_DIST)
                for p in tr['peaks']:
                    kk = (f, int(p[0]), int(p[1]))
                    if kk not in cand:
                        need.append(kk)
            if need:
                ir_rgb, _, _, _ = _letterbox(cv2.cvtColor(cv2.imread(str(
                    ROOT / split / inst / 'ir' / 'search' / f'{f:06d}.jpg')),
                    cv2.COLOR_BGR2RGB), IMG_SIZE)
                ir_bgr = cv2.cvtColor(ir_rgb, cv2.COLOR_RGB2BGR)
                crops = [crop_patch(ir_bgr, kk[1], kk[2]) for kk in need]
                clips = clip_embed(crops)
                gaps = []
                for s in range(0, len(crops), 64):
                    xs = torch.stack([norm224(c) for c in crops[s:s + 64]]).to(DEVICE)
                    gaps.append(enc.intermediate_gap(xs).cpu())
                gaps = torch.cat(gaps)
                for j, kk in enumerate(need):
                    cand[kk] = {'gap': gaps[j], 'clip': torch.from_numpy(clips[j])}
            for tr in ftrs:
                labs = label_peaks(tr['peaks'], fr['gt_box'], fr['distractor_boxes'])
                lines.append(json.dumps({
                    'split': split, 'inst': inst, 't': tr['t'], 'f': f,
                    'diff': fr['difficulty'],
                    'gt': [round(float(fr['gt_center'][0]), 2), round(float(fr['gt_center'][1]), 2)],
                    'obj': [round(float(fr['obj_size'][0]), 2), round(float(fr['obj_size'][1]), 2)],
                    'gt_box': [round(float(v), 2) for v in fr['gt_box']],
                    'd_boxes': [[round(float(v), 2) for v in b] for b in fr['distractor_boxes']],
                    'peaks': [[int(p[0]), int(p[1]), round(float(p[2]), 6)] for p in tr['peaks']],
                    'labels': labs,
                    'n_pos': sum(1 for v in labs if v == 2),
                    'cover': any(v == 2 for v in labs)}, ensure_ascii=False))
        torch.save(cand, shard)
        n_cand = len(cand)
        del cand
        gc.collect()
        out_f.write('\n'.join(lines) + ('\n' if lines else ''))
        out_f.flush()
        done.add(inst)
        prog_path.write_text(json.dumps(sorted(done)), encoding='utf-8')
        print(f'[{split}{suffix}] {inst}: {len(lines)} trials, cand={n_cand} '
              f'（{time.time() - t0:.0f}s / {len(done)}/{len(by_inst)}）', flush=True)
    out_f.close()
    del model, enc
    torch.cuda.empty_cache()


def phase_check():
    assert_env()
    n_train_t = sum(len(t) for _i, t, _f in instances('train'))
    n_train_f = sum(len(f) for _i, _t, f in instances('train'))
    n_val_t = sum(len(t) for _i, t, _f in instances('val'))
    print(f'计数: train 模板 {n_train_t} / 帧 {n_train_f} / val 模板 {n_val_t}')
    assert (n_train_t, n_train_f, n_val_t) == (4386, 5880, 1355)
    tr = sample_trials()
    print(f'采样断言通过: train={len(tr["train"])} val={len(tr["val"])}（seed 42 单流）')
    # G0：复刻 n_e116b_s42_trials.jsonl 的 top-5 峰（3 实例 × 2 模板 × 2 帧）
    ref = {}
    with open(TRIALS_REF, encoding='utf-8') as fh:
        for ln in fh:
            r = json.loads(ln)
            ref[(r['inst'], r['t'], r['f'])] = r
    tcache = torch.load(CACHE, map_location='cpu')
    model = build_locator()
    bad = 0
    n_chk = 0
    for inst in sorted({k[0] for k in ref})[:3]:
        keys = sorted(k for k in ref if k[0] == inst)[:4]
        for k in keys:
            inst_, t, f = k
            fr = load_frame(ROOT / 'val' / inst_, f)
            logits = n.forward_cached(model, True, fr['ir'].unsqueeze(0).to(DEVICE),
                                      [tcache[inst_][str(t)]])
            hn = torch.sigmoid(logits)[0, 0].cpu().numpy()
            peaks = nms_peaks(hn, K, MIN_DIST)
            rp = ref[k]['peaks']
            n_chk += 1
            for i in range(min(5, len(rp))):
                dx = abs(peaks[i][0] - rp[i][0]) + abs(peaks[i][1] - rp[i][1])
                dv = abs(peaks[i][2] - rp[i][2])
                if dx != 0 or dv > 1e-5:
                    bad += 1
                    print(f'  G0 不符 {k} peak{i}: 我={peaks[i]} 参考={rp[i]}')
    print(f'G0: {n_chk} trials × top-5 峰复刻，不符 {bad} 处')
    assert bad == 0, 'G0 复刻失败，停机'
    print('CHECK PASS')


def phase_smoke():
    assert_env()
    tr = sample_trials()
    enc = DINOv2TemplateEncoder(pretrained=True).to(DEVICE).eval()
    clip_embed = build_clip()
    picks = {'train': sorted({r['inst'] for r in tr['train']})[:2],
             'val': sorted({r['inst'] for r in tr['val']})[:1]}
    for split, insts in picks.items():
        tc = build_tmpl_cache(split, insts, enc, clip_embed)
        gen_split(split, [r for r in tr[split] if r['inst'] in insts], tc, suffix='_smoke')
        for ln in open(OUT_DIR / f'cmi_probe_trials_{split}_smoke.jsonl', encoding='utf-8'):
            r = json.loads(ln)
            if r['inst'] not in insts:
                continue
        # 统计
        recs = [json.loads(l) for l in open(OUT_DIR / f'cmi_probe_trials_{split}_smoke.jsonl',
                                            encoding='utf-8')]
        recs = [r for r in recs if r['inst'] in insts]
        cov = sum(r['cover'] for r in recs)
        hist = [0, 0, 0]
        for r in recs:
            for v in r['labels']:
                hist[v] += 1
        print(f'[smoke {split}] trials={len(recs)} 覆盖={cov}（{cov / len(recs):.3f}）'
              f' 标签 易/难/正={hist[0]}/{hist[1]}/{hist[2]}')
    print('SMOKE DONE')


def phase_full():
    assert_env()
    tr = sample_trials()
    enc = DINOv2TemplateEncoder(pretrained=True).to(DEVICE).eval()
    clip_embed = build_clip()
    for split in ('train', 'val'):
        insts = sorted({r['inst'] for r in tr[split]})
        tc = build_tmpl_cache(split, insts, enc, clip_embed)
        gen_split(split, tr[split], tc)
        del tc
        torch.cuda.empty_cache()
    print('FULL DONE')


def phase_manifest():
    lines = []
    for split in ('train', 'val'):
        p = OUT_DIR / f'cmi_probe_trials_{split}.jsonl'
        recs = [json.loads(l) for l in open(p, encoding='utf-8')]
        cov = [r for r in recs if r['cover']]
        ph = [r for r in recs if r['cover'] and 1 in r['labels']]
        hist = [0, 0, 0]
        for r in recs:
            for v in r['labels']:
                hist[v] += 1
        lines.append({'type': 'trials', 'split': split, 'file': p.name,
                      'sha256': sha256(p), 'n_trials': len(recs),
                      'cover_trials': len(cov), 'cover_rate': round(len(cov) / len(recs), 4),
                      'pairable_trials(正且难负)': len(ph),
                      'label_hist_易难正': hist})
    for p in sorted(EXT.glob('cmi_probe_*.pt')):
        lines.append({'type': 'cache', 'file': str(p), 'bytes': p.stat().st_size})
    lines.append({'type': 'env', 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                  'seed': SEED, 'K': K, 'min_dist': MIN_DIST})
    mp = OUT_DIR / 'cmi_probe_manifest.jsonl'
    with open(mp, 'w', encoding='utf-8') as f:
        f.write('\n'.join(json.dumps(v, ensure_ascii=False) for v in lines) + '\n')
    for v in lines:
        print(json.dumps(v, ensure_ascii=False))
    print(f'MANIFEST → {mp.name}')


def phase_conv():
    """把已建合并 tmpl 缓存切成按实例分片（mmap 读，避免 13.8GB 全驻 12GB 内存）。"""
    for split in ('train', 'val'):
        big = EXT / f'cmi_probe_tmpl_{split}.pt'
        if not big.exists():
            continue
        data = torch.load(big, map_location='cpu', mmap=True)
        for inst in data:
            shard = EXT / f'cmi_probe_tmpl_{split}_{inst}.pt'
            if shard.exists():
                continue
            d = {t: {k: (v.clone() if isinstance(v, torch.Tensor) else v)
                     for k, v in e.items()} for t, e in data[inst].items()}
            n_t = len(d)
            torch.save(d, shard)
            del d
            gc.collect()
            print(f'  [conv] {split}/{inst}: {n_t} 模板', flush=True)
        del data
        gc.collect()
        big.unlink()
        print(f'[conv] {split} 完成，合并文件已删', flush=True)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', required=True,
                    choices=['check', 'smoke', 'full', 'manifest', 'conv'])
    a = ap.parse_args()
    EXT.mkdir(parents=True, exist_ok=True)
    {'check': phase_check, 'smoke': phase_smoke,
     'full': phase_full, 'manifest': phase_manifest, 'conv': phase_conv}[a.phase]()


if __name__ == '__main__':
    main()
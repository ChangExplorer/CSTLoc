# baselines/xmodal_matcher/run.py —— 跨模态匹配器臂（RoMa 官方权重，任务书 2.4 臂 A）
#
# 每对（模板, 场景帧）做 RoMa 稠密匹配 → 高置信匹配点置信度加权质心（场景侧）
# 为预测点；坐标换算回原始 1920×1080（RoMa 输出在 upsample_res 空间）。
# 同源自检：同图匹配质心须落在图中心（±2% 边长）。
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'references' / 'repos' / 'RoMa'))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from romatch import roma_outdoor  # noqa: E402

DATA = Path(os.environ['CSTLOC_DATA_ROOT'])
MIN_GAP = 30
DEVICE = torch.device('cuda')


def centroid_and_peaks(model, im_A, im_B):
    """RoMa 匹配 → (场景侧加权质心(原图坐标), top-32 匹配点(原图坐标, 置信度))。"""
    warp, certainty = model.match(im_A, im_B, device=DEVICE)
    matches, cert = model.sample(warp, certainty, num=4000)
    matches = matches.cpu().numpy()
    cert = cert.cpu().numpy()
    # RoMa 输出为归一化坐标 [-1,1]（场景侧 xB,yB），映射回原图 1920×1080
    order = np.argsort(-cert)
    matches, cert = matches[order], cert[order]
    top = matches[:200]
    tc = cert[:200]
    if len(tc) == 0 or tc.sum() < 1e-8:
        return [960.0, 540.0], []
    cx = float((top[:, 2] * tc).sum() / tc.sum())
    cy = float((top[:, 3] * tc).sum() / tc.sum())
    peaks = [[float((x + 1) / 2 * 1920), float((y + 1) / 2 * 1080), float(c)]
             for (x, y), c in zip(matches[:32][:, 2:], cert[:32])]
    return [(cx + 1) / 2 * 1920, (cy + 1) / 2 * 1080], peaks


def main():
    ap = __import__('argparse').ArgumentParser()
    ap.add_argument('--split', default='test', choices=['test'])
    ap.add_argument('--instances', default=None)
    ap.add_argument('--selfcheck', action='store_true')
    args = ap.parse_args()
    torch.manual_seed(0)
    model = roma_outdoor(device=DEVICE, coarse_res=560, upsample_res=(864, 1152),
                         use_custom_corr=False)
    here = Path(os.environ.get('XMODAL_OUT_DIR', str(Path(__file__).resolve().parent)))

    if args.selfcheck:
        im = str(DATA / 'test' / 'car_001' / 'ir' / 'search' / '000038.jpg')
        c, pk = centroid_and_peaks(model, im, im)
        print('同图自检质心:', c, '（期望 ≈[960,540]）')
        return

    prog_f = here / 'roma_progress.json'
    done = json.loads(prog_f.read_text()) if prog_f.exists() else {}
    # 断点续传：以 results.jsonl 实有记录为准（双进程共享进度文件会互相覆写，不可信）
    res_path = here / 'results.jsonl'
    done_trials = set()
    if res_path.exists():
        for _l in open(res_path, encoding='utf-8'):
            try:
                _r = json.loads(_l)
                done_trials.add((_r['inst'], _r['t'], _r['f']))
            except Exception:
                pass
    res_f = open(here / 'results.jsonl', 'a', encoding='utf-8')
    pk_f = open(here / 'peaks.jsonl', 'a', encoding='utf-8')
    t0 = time.time()
    for inst_dir in sorted((DATA / args.split).iterdir()):
        inst = inst_dir.name
        if args.instances and inst not in args.instances.split(','):
            continue
        if done.get(inst):
            continue
        tmpls = sorted(int(p.stem) for p in (inst_dir / 'rgb' / 'template').glob('*.jpg'))
        frames = sorted(int(p.stem) for p in (inst_dir / 'ir' / 'search').glob('*.json'))
        lines = []
        for t in tmpls:
            im_A = str(inst_dir / 'rgb' / 'template' / f'{t:06d}.jpg')
            for f in frames:
                if abs(f - t) < MIN_GAP:
                    continue
                if (inst, t, f) in done_trials:
                    continue
                im_B = str(inst_dir / 'ir' / 'search' / f'{f:06d}.jpg')
                pred, peaks = centroid_and_peaks(model, im_A, im_B)
                lines.append(json.dumps({'inst': inst, 'tmpl_mod': 'rgb', 't': t,
                                         'scene_mod': 'ir', 'f': f, 'pred': pred},
                                        ensure_ascii=False))
                pk_f.write(json.dumps({'inst': inst, 't': t, 'f': f, 'peaks': peaks},
                                      ensure_ascii=False) + '\n')
            res_f.write('\n'.join(lines) + ('\n' if lines else ''))
            res_f.flush()
            lines = []
        done[inst] = 1
        prog_f.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding='utf-8')
        print(f'[{inst}] 完成（累计 {time.time() - t0:.0f}s）', flush=True)
    res_f.close()
    pk_f.close()


if __name__ == '__main__':
    main()

# evaluation/adapters/tcia_to_results.py —— TCIA 逐实例分数 → 三层契约标准格式
#
# 输入：TCIA 评测落盘的逐实例 jsonl，每行至少含:
#   {"inst","t","f","pred_lb":[x,y]} 或 {"inst","t","f","peaks":[[x,y,score],...],"tcia":[score,...]}
#   peaks / pred_lb 为 letterbox 384×640 坐标（1920×1080 等比 ÷3、上下补 12px）。
# 输出：
#   results.jsonl: {"inst","tmpl_mod":"rgb","t","scene_mod":"ir","f","pred":[x,y]}（原图坐标）
#   peaks.jsonl  : {"inst","t","f","peaks":[[x,y,score],...]}（原图坐标，供 oracle@K）
# GT/难度/类别不写入，由 evaluation/aggregate.py 从数据集挂接。
import argparse, json, glob
from pathlib import Path

SCALE = 3.0
PAD_Y = 12.0

def lb2orig(p):
    return [float(p[0]) * SCALE, (float(p[1]) - PAD_Y) * SCALE]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scores-dir', required=True, help='逐实例 jsonl 目录')
    ap.add_argument('--out-results', required=True)
    ap.add_argument('--out-peaks', default=None)
    ap.add_argument('--coord', default='letterbox', choices=['letterbox','orig'])
    args = ap.parse_args()
    conv = lb2orig if args.coord == 'letterbox' else (lambda p: [float(p[0]), float(p[1])])
    paths = sorted(glob.glob(str(Path(args.scores_dir) / '*.jsonl')))
    assert paths, args.scores_dir
    n = npk = 0
    pk_f = open(args.out_peaks, 'w', encoding='utf-8') if args.out_peaks else None
    with open(args.out_results, 'w', encoding='utf-8') as out:
        for p in paths:
            for line in open(p, encoding='utf-8'):
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if 'pred_lb' in r:
                    pred_lb = r['pred_lb']
                else:
                    assert 'peaks' in r and 'tcia' in r, (p, r.get('keys'))
                    i = int(max(range(len(r['tcia'])), key=lambda j: r['tcia'][j]))
                    pred_lb = r['peaks'][i][:2]
                out.write(json.dumps({'inst': r['inst'], 'tmpl_mod': 'rgb', 't': int(r['t']),
                                      'scene_mod': 'ir', 'f': int(r['f']),
                                      'pred': conv(pred_lb)}, ensure_ascii=False) + '\n')
                n += 1
                if pk_f is not None and r.get('peaks'):
                    peaks = [conv(q[:2]) + [float(q[2])] for q in r['peaks']]
                    pk_f.write(json.dumps({'inst': r['inst'], 't': int(r['t']),
                                           'f': int(r['f']), 'peaks': peaks},
                                          ensure_ascii=False) + '\n')
                    npk += 1
    if pk_f is not None:
        pk_f.close()
    print(f'{args.scores_dir} → {args.out_results}: {n} 行' + (f'; peaks {npk} 行' if args.out_peaks else ''))

if __name__ == '__main__':
    main()

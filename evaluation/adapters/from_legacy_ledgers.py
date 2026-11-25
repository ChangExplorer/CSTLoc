# evaluation/adapters/from_legacy_ledgers.py —— 旧底稿 → 三层契约标准格式
#
# 把 code/diag_out/ 下既有 trials / peaks32 底稿转换为 results.jsonl（+ 可选 peaks.jsonl）。
# 坐标口径：旧底稿（n_*/e130_*）的 pred/base/peaks 为 **letterbox 384×640 坐标**
#（原图 1920×1080 等比 ÷3 后上下各补 12px），须转回原图坐标：`x*3, (y−12)*3`。
import argparse
import json

LB_SCALE = 3.0      # 1920→640
LB_PAD_Y = 12.0     # (384−360)/2


def lb2orig(p):
    return [p[0] * LB_SCALE, (p[1] - LB_PAD_Y) * LB_SCALE]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='旧底稿 jsonl 路径')
    ap.add_argument('--pred-field', required=True, choices=['base', 'pred'],
                    help='坐标取自哪个字段（裸定位器=base，验证器重排=pred）')
    ap.add_argument('--out', required=True, help='输出 results.jsonl')
    ap.add_argument('--peaks-out', default=None, help='底稿含 peaks 字段时可同时导出 peaks.jsonl')
    ap.add_argument('--coords', default='letterbox', choices=['letterbox', 'orig'],
                    help='底稿坐标系（letterbox 会转回原图坐标）')
    args = ap.parse_args()

    conv = lb2orig if args.coords == 'letterbox' else (lambda p: [float(p[0]), float(p[1])])
    n = n_pk = 0
    pk_f = open(args.peaks_out, 'w', encoding='utf-8') if args.peaks_out else None
    with open(args.out, 'w', encoding='utf-8') as out_f:
        for line in open(args.src, encoding='utf-8'):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pred = conv([float(r[args.pred_field][0]), float(r[args.pred_field][1])])
            out_f.write(json.dumps({'inst': r['inst'], 'tmpl_mod': 'rgb', 't': int(r['t']),
                                    'scene_mod': 'ir', 'f': int(r['f']), 'pred': pred},
                                   ensure_ascii=False) + '\n')
            n += 1
            if pk_f and r.get('peaks'):
                peaks = [conv([float(p[0]), float(p[1])]) + [float(p[2])] for p in r['peaks']]
                pk_f.write(json.dumps({'inst': r['inst'], 't': int(r['t']), 'f': int(r['f']),
                                       'peaks': peaks}, ensure_ascii=False) + '\n')
                n_pk += 1
    if pk_f:
        pk_f.close()
    print(f'{args.src} → {args.out}: {n} 行' + (f'；peaks {n_pk} 行' if pk_f else ''))


if __name__ == '__main__':
    main()

# 最终 v3 test 评测入口：冻结发布定位器 + TCIA best_fullval，输出 summary JSON。
# 由 scripts/05_eval_test.sh 调用（设置 TCIA_BEST_CKPT / EVAL_TAG / EVAL_OUT）。
import json
import os
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))


def main():
    import torch
    from models.tcia import TCIAv2
    from tools.tcia_eval import run_split

    tag = os.environ.get('EVAL_TAG', 'final')
    ck = Path(os.environ['TCIA_BEST_CKPT'])
    st = torch.load(ck, map_location='cpu')
    model = TCIAv2().to('cuda').eval()
    model.load_adapter_state_dict(st['state'])
    summary, rows = run_split(model, 'test', write_scores=False)
    out = Path(os.environ.get(
        'EVAL_OUT', str(CODE.parent / 'evaluation' / 'results' / f'tcia_{tag}_test_summary.json')))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print('SAVED', out, flush=True)


if __name__ == '__main__':
    main()

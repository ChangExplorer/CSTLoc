# 消融 TCIA test wrapper：使用环境指定的定位器 + TCIA best_fullval
import os, json, sys
from pathlib import Path
CODE=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(CODE))
import torch
from models.tcia_v2 import TCIAv2
from tools.tcia_v2_select_fullval import run_split
tag=os.environ['ABL_TAG']; ck=Path(os.environ['TCIA_BEST_CKPT'])
st=torch.load(ck,map_location='cpu')
model=TCIAv2().to('cuda').eval(); model.load_adapter_state_dict(st['state'])
summary,rows=run_split(model,'test',write_scores=False,staging_dir=os.environ.get('CSTLOC_STAGING'))
out=CODE.parent/'evaluation'/'cache'/'ablation'/f'tcia_{tag}_test_summary.json'; out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(summary,ensure_ascii=False,indent=1),encoding='utf-8')
print('SAVED',out,flush=True)

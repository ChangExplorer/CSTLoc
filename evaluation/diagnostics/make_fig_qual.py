# -*- coding: utf-8 -*-
import json, os
from pathlib import Path
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

BASE=Path(os.environ.get('CSTLOC_ROOT', Path(__file__).resolve().parents[2]))
DATA=Path(os.environ.get('CSTLOC_DATA_ROOT', '/home/user/Code/Datasets/CSTLoc'))
OUT=BASE/'evaluation/diagnostics/figures/fig_qual.pdf'
plt.rcParams.update({'font.size':6.2,'font.family':'sans-serif','font.sans-serif':['Arial','Helvetica','DejaVu Sans'],'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42})

SYS={'PerSAM':'evaluation/cache/v3_comparison/persam.results.jsonl',
     'NCC':'evaluation/cache/v3_comparison/ncc.results.jsonl',
     'RoMa':'evaluation/cache/v3_comparison/roma.results.jsonl',
     'DINOv2':'evaluation/cache/v3_comparison/dinov2.results.jsonl',
     'Grounding':'evaluation/cache/v3_comparison/grounding.results.jsonl',
     'Ours':'evaluation/cache/v3_test_diag/main_results/results_full_s2024.jsonl'}
PRED={}
for k,rel in SYS.items():
    d={}
    for ln in open(BASE/rel,encoding='utf-8'):
        r=json.loads(ln); d[(r['inst'],int(r['t']),int(r['f']))]=np.array(r['pred'],float)
    PRED[k]=d

ROWS=[('truck_001',660,3990),('pedestrian_160',300,5010),('car_103',6180,5610)]
COLS=['Template','PerSAM','NCC','RoMa','DINOv2','Grounding','GT','Ours']
LABELS={'Template':'Template','PerSAM':'PerSAM','NCC':'NCC','RoMa':'RoMa','DINOv2':'DINOv2','Grounding':'GDINO','GT':'GT','Ours':'Ours'}
W=[0.50,1,1,1,1,1,1,1]

def load_row(inst,t,f):
    jp=DATA/'test'/inst/'ir'/'search'/f'{f:06d}.json'
    d=json.load(open(jp,encoding='utf-8'))
    tbox=None; dboxes=[]
    for s in d['shapes']:
        pts=np.asarray(s['points'],float)
        b=[pts[:,0].min(),pts[:,1].min(),pts[:,0].max(),pts[:,1].max()]
        if (s.get('flags') or {}).get('target'): tbox=b
        else: dboxes.append(b)
    img=cv2.cvtColor(cv2.imread(str(jp.with_suffix('.jpg'))),cv2.COLOR_BGR2RGB)
    tp=DATA/'test'/inst/'rgb'/'template'/f'{t:06d}.jpg'
    timg=cv2.cvtColor(cv2.imread(str(tp)),cv2.COLOR_BGR2RGB)
    return img,timg,tbox,dboxes

fig=plt.figure(figsize=(7.2,3.25))
gs=fig.add_gridspec(3,8,width_ratios=W,wspace=0.025,hspace=0.30,left=0.006,right=0.996,top=0.88,bottom=0.115)
template_axes=[]
for ri,(inst,t,f) in enumerate(ROWS):
    img,timg,tbox,dboxes=load_row(inst,t,f)
    key=(inst,t,f)
    for ci,col in enumerate(COLS):
        ax=fig.add_subplot(gs[ri,ci])
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color('0.75'); sp.set_linewidth(0.3)
        if col=='Template':
            ax.imshow(timg); ax.set_title(LABELS[col],pad=2.5,fontsize=6.2)
            template_axes.append(ax)
            continue
        ax.imshow(img)
        ax.set_title(LABELS[col],pad=2.5,fontsize=6.2)
        ax.add_patch(Rectangle((tbox[0],tbox[1]),tbox[2]-tbox[0],tbox[3]-tbox[1],
                               fill=False,edgecolor='#19c819',linewidth=0.65,zorder=5))
        for b in dboxes:
            ax.add_patch(Rectangle((b[0],b[1]),b[2]-b[0],b[3]-b[1],fill=False,
                                   edgecolor='#ff9900',linewidth=0.5,linestyle=(0,(4,2)),zorder=5))
        if col!='GT':
            p=PRED[col][key]
            ax.plot([p[0]],[p[1]],marker='+',ms=4.5,color='#ff1f1f',mew=0.7,zorder=8)
        ax.set_xlim(0,img.shape[1]); ax.set_ylim(img.shape[0],0)

handles=[Line2D([0],[0],color='#19c819',lw=0.65,label='Target instance'),
         Line2D([0],[0],color='#ff9900',lw=0.5,ls=(0,(4,2)),label='Same-class distractor'),
         Line2D([0],[0],marker='+',linestyle='None',color='#ff1f1f',mew=0.7,ms=5,label='Prediction')]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=6.0,
           handlelength=1.5,columnspacing=1.4,bbox_to_anchor=(0.5,0.002))

# Audit the final rendered axes geometry before writing the deliverables.
import sys
sys.path.insert(0,'/mnt/c/Users/Administrator/.dsh/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment
require_matplotlib_panel_alignment(
    fig,
    json_out='/mnt/c/Users/Public/fig_qual.alignment.json',
    overlay_svg='/mnt/c/Users/Public/fig_qual.alignment.svg',
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    exclude_axes=template_axes,  # Reference templates have different aspect ratios; audit the comparable IR scene panels.
    strict=True,
)

OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,dpi=400)
print('WROTE',OUT)
for inst,t,f in ROWS:
    img,timg,tbox,dboxes=load_row(inst,t,f)
    area=(tbox[2]-tbox[0])*(tbox[3]-tbox[1])/(img.shape[1]*img.shape[0])
    print(inst,t,f,'K',len(dboxes),'area',round(area,4))
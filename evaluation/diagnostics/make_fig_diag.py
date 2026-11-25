# -*- coding: utf-8 -*-
import json, os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

BASE=Path(os.environ.get('CSTLOC_ROOT', Path(__file__).resolve().parents[2]))
d=json.load(open(BASE/'evaluation/cache/diagnostics_v3/diagnostics_v3.json',encoding='utf-8'))
S=d['systems']
OUT=BASE/'evaluation/diagnostics/figures/fig_diag.pdf'
plt.rcParams.update({'font.size':8.0,'font.family':'DejaVu Sans','pdf.fonttype':42,'ps.fonttype':42,
                     'axes.linewidth':0.6,'xtick.major.width':0.6,'ytick.major.width':0.6})

order=['Full','Locator','Grounding DINO','DINOv2','RoMa','NCC','PerSAM']
colors={'Full':'#d62728','Locator':'#1f77b4','Grounding DINO':'#2ca02c','DINOv2':'#ff7f0e',
        'RoMa':'#9467bd','NCC':'#8c564b','PerSAM':'#7f7f7f'}
styles={'Full':'-','Locator':'--'}
DISP={'Grounding DINO':'GDINO'}
label_y={'Full':0.500,'Locator':0.341,'Grounding DINO':0.205,'DINOv2':0.178,
         'RoMa':0.150,'PerSAM':0.048,'NCC':0.076}

fig=plt.figure(figsize=(7.2,2.55))
gs=GridSpec(1,2,width_ratios=[1.62,1.0],wspace=0.34,left=0.075,right=0.985,top=0.90,bottom=0.17)

ax=fig.add_subplot(gs[0,0])
for name in order:
    tc=S[name]['time_curve']
    x=[t['rho_mid'] for t in tc]; y=[t['pm'] for t in tc]
    lw=1.8 if name in ('Full','Locator') else 1.0
    ax.plot(x,y,color=colors[name],lw=lw,ls=styles.get(name,'-'),marker='o',
            ms=2.0 if name in ('Full','Locator') else 1.4,mec='none',
            zorder=5 if name=='Full' else 3)
    ax.text(1.01,label_y[name],DISP.get(name,name),color=colors[name],fontsize=7.0,va='center',ha='left')
ax.set_xlim(0,1.42); ax.set_ylim(0,0.62)
ax.set_xticks(np.arange(0,1.01,0.2))
ax.set_xlabel('Normalized time interval  $|f-t|/\\max|f-t|$',labelpad=2)
ax.set_ylabel('PCK@$0.5\\cdot\\max$ (pair-macro)',labelpad=2)
ax.grid(True,ls=':',lw=0.4,color='0.75')
ax.set_title('(a) Time-interval effect',loc='left',fontsize=8.4,pad=4)

ax2=fig.add_subplot(gs[0,1])
ys=np.arange(len(order))[::-1]
inst=[S[n]['fail_instance_hard'] for n in order]
bg=[S[n]['fail_background_hard'] for n in order]
ax2.barh(ys,inst,color='#2c7fb8',height=0.62)
ax2.barh(ys,bg,left=inst,color='#f4a261',height=0.62)
for y,i,b,n in zip(ys,inst,bg,order):
    if n=='Full':
        continue
    if i>=0.18:
        ax2.text(i/2,y,f'{i:.2f}',ha='center',va='center',fontsize=7.2,color='white')
    else:
        ax2.text(i+0.015,y,f'{i:.2f}',ha='left',va='center',fontsize=7.0,color='#1a5276')
# Full 条内直接标注两类；不另设图例
y0=ys[0]; i0=inst[0]
ax2.text(i0/2,y0,f'{i0:.2f} instance',ha='center',va='center',fontsize=6.4,color='white')
ax2.text(i0+(1-i0)/2,y0,f'{1-i0:.2f} bg',ha='center',va='center',fontsize=6.4,color='#222222')
ax2.set_yticks(ys); ax2.set_yticklabels([DISP.get(n,n) for n in order],fontsize=7.4)
ax2.set_xlim(0,1.0); ax2.set_ylim(-0.6,len(order)-0.4)
ax2.set_xlabel('Share of failed hard trials',labelpad=2)
ax2.grid(True,axis='x',ls=':',lw=0.4,color='0.75')
ax2.set_title('(b) Failure composition on hard',loc='left',fontsize=8.4,pad=4)
for s in ('top','right'):
    ax2.spines[s].set_visible(False)
ax2.tick_params(axis='y',length=0)

OUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUT,dpi=400)
print('WROTE',OUT)

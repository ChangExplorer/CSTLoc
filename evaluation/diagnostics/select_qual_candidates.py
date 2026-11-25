#!/usr/bin/env python
import json, os
from pathlib import Path
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

BASE=Path(os.environ.get('CSTLOC_ROOT', Path(__file__).resolve().parents[2])); DATA=Path(os.environ.get('CSTLOC_DATA_ROOT', '/home/user/Code/Datasets/CSTLoc'))
OUT=BASE/'evaluation/cache/diagnostics_v3'; OUT.mkdir(parents=True, exist_ok=True)
SYS={
 'PerSAM':'evaluation/cache/v3_comparison/persam.results.jsonl',
 'NCC':'evaluation/cache/v3_comparison/ncc.results.jsonl',
 'RoMa':'evaluation/cache/v3_comparison/roma.results.jsonl',
 'DINOv2':'evaluation/cache/v3_comparison/dinov2.results.jsonl',
 'Grounding':'evaluation/cache/v3_comparison/grounding.results.jsonl',
 'Locator':'evaluation/cache/v3_test_diag/main_results/results_locator_s2024.jsonl',
 'Full':'evaluation/cache/v3_test_diag/main_results/results_full_s2024.jsonl'}

def frames():
    F={}
    for inst_dir in sorted((DATA/'test').iterdir()):
        if not inst_dir.is_dir(): continue
        inst=inst_dir.name
        for jf in sorted((inst_dir/'ir'/'search').glob('*.json')):
            d=json.load(open(jf,encoding='utf-8'))
            tbox=None; db=[]; dc=[]
            for s in d.get('shapes',[]):
                p=np.asarray(s.get('points',[]),float)
                if p.size==0: continue
                b=[float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]
                c=[(b[0]+b[2])/2,(b[1]+b[3])/2]
                if (s.get('flags') or {}).get('target'): tbox=(c,[b[2]-b[0],b[3]-b[1]],b)
                else: db.append(b); dc.append(c)
            if tbox is None: continue
            F[(inst,int(jf.stem))]={'gt':tbox[0],'obj':tbox[1],'tbox':tbox[2],
                                    'dboxes':db,'dc':dc,'hard':bool(db),
                                    'w':d.get('imageWidth',1920),'h':d.get('imageHeight',1080)}
    return F

F=frames()
def load(path):
    P={}
    for ln in open(BASE/path,encoding='utf-8'):
        r=json.loads(ln); P[(r['inst'],int(r['t']),int(r['f']))]=np.array(r['pred'],float)
    return P
P={k:load(v) for k,v in SYS.items()}

def err(p,fr):
    return float(np.hypot(*(p-fr['gt'])))
def hit(p,fr):
    return err(p,fr) < 0.5*max(fr['obj'])

# fixed pair
fixed=('truck_001',660,3990)
print('FIXED', fixed, {k:(round(err(P[k][fixed],F[(fixed[0],fixed[2])]),1),hit(P[k][fixed],F[(fixed[0],fixed[2])])) for k in SYS})

cands={'pedestrian':[], 'car':[]}
keys=list(P['Full'].keys())
for key in keys:
    inst,t,f=key; cls=inst.rsplit('_',1)[0]
    if cls not in cands: continue
    fr=F[(inst,f)]
    K=len(fr['dboxes'])
    if not fr['hard'] or K<3 or K>7: continue
    pfull=P['Full'][key]
    if not hit(pfull,fr): continue
    # target area ratio and clarity
    area=(fr['obj'][0]*fr['obj'][1])/(fr['w']*fr['h'])
    if area < 0.00035: continue
    efull=err(pfull,fr)/max(fr['obj'])
    misses=sum(not hit(P[k][key],fr) for k in ['PerSAM','NCC','RoMa','DINOv2','Grounding'])
    errs={k:err(P[k][key],fr)/max(fr['obj']) for k in SYS}
    # margin vs best baseline (grounding often best)
    best_base=min(errs[k] for k in ['PerSAM','NCC','RoMa','DINOv2','Grounding'])
    margin=best_base-efull
    cands[cls].append(dict(inst=inst,t=t,f=f,K=K,area=area,efull=efull,misses=misses,
                           margin=margin,errs=errs))
for cls in cands:
    cands[cls].sort(key=lambda d:(-d['misses'],-d['margin'],d['efull'],-d['area']))
    print('\n',cls,len(cands[cls]))
    for d in cands[cls][:15]:
        print(d['inst'],d['t'],d['f'],'K',d['K'],'miss',d['misses'],'efull',round(d['efull'],3),
              'margin',round(d['margin'],3),'area',round(d['area']*1e4,2),'errs',{k:round(v,2) for k,v in d['errs'].items()})
json.dump(cands,open(OUT/'qual_candidates.json','w'),ensure_ascii=False,indent=1)

# contact sheets
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
def sheet(cls, items, path, cols=4):
    tw,th=480,270; rows=(len(items)+cols-1)//cols
    canvas=Image.new('RGB',(cols*tw,rows*(th+40)),'white'); dr=ImageDraw.Draw(canvas)
    for i,d in enumerate(items):
        inst,t,f=d['inst'],d['t'],d['f']; fr=F[(inst,f)]
        img=cv2.imread(str(DATA/'test'/inst/'ir'/'search'/f'{f:06d}.jpg'))
        img=cv2.resize(img,(tw,th))
        sx,sy=tw/fr['w'],th/fr['h']
        def box(b,col):
            cv2.rectangle(img,(int(b[0]*sx),int(b[1]*sy)),(int(b[2]*sx),int(b[3]*sy)),col,2)
        box(fr['tbox'],(0,255,0))
        for b in fr['dboxes']: box(b,(0,165,255))
        p=P['Full'][(inst,t,f)]
        cv2.drawMarker(img,(int(p[0]*sx),int(p[1]*sy)),(255,0,255),cv2.MARKER_CROSS,16,2)
        canvas.paste(Image.fromarray(cv2.cvtColor(img,cv2.COLOR_BGR2RGB)),((i%cols)*tw,(i//cols)*(th+40)))
        txt=f"{inst} t={t} f={f} K={d['K']} miss={d['misses']} e={d['efull']:.2f}"
        dr.text(((i%cols)*tw+4,(i//cols)*(th+40)+th+6),txt,fill='black',font=font)
    canvas.save(path); print('saved',path)
sheet('pedestrian',cands['pedestrian'][:12],OUT/'qual_sheet_pedestrian.jpg')
sheet('car',cands['car'][:12],OUT/'qual_sheet_car.jpg')

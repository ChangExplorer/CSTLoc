# PerSAM C 臂：RGB 模板(参考图) -> IR 场景，mask 质心作为预测点
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np, cv2, torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
REPO = os.environ.get("CSTLOC_PERSAM_REPO", str(ROOT / "references" / "repos" / "Personalize-SAM"))
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, REPO); os.chdir(REPO)
from per_segment_anything import sam_model_registry, SamPredictor  # noqa

DATA = os.environ["CSTLOC_DATA_ROOT"]
SAM_CKPT = os.environ.get("CSTLOC_SAM_CKPT", str(ROOT / "references" / "weights" / "sam_vit_h_4b8939.pth"))
CLS = {"pedestrian":"pedestrian","car":"car","tricycle":"tricycle","elebike":"electric bike",
       "bus":"bus","bike":"bike","animal":"animal","truck":"truck"}
DEV = "cuda"

def point_selection(mask_sim, topk=1):
    h, w = mask_sim.shape  # 注意 persam 原实现 w,h=mask_sim.shape 后按 h 走，保持等价
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
    last_xy = mask_sim.flatten(0).topk(topk, largest=False)[1]
    last_x = (last_xy // h).unsqueeze(0)
    last_y = (last_xy - last_x * h)
    last_xy = torch.cat((last_y, last_x), dim=0).permute(1, 0)
    last_label = np.array([0] * topk)
    last_xy = last_xy.cpu().numpy()
    return topk_xy, topk_label, last_xy, last_label

def build_predictor():
    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT)
    sam.to(device=DEV); sam.eval()
    return SamPredictor(sam)

def ref_embedding(predictor, tmpl_path):
    img = cv2.cvtColor(cv2.imread(str(tmpl_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    mask = np.zeros((h, w, 3), np.uint8)
    m0 = int(round(0.10 * h)); n0 = int(round(0.10 * w))
    mask[m0:h - m0, n0:w - n0] = 255   # 模板 margin=1.2 -> 目标约居中 83%，取中心 80% 作为参考 mask
    ref_mask = predictor.set_image(img, mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)
    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0:2], mode="bilinear").squeeze()[0]
    tgt = ref_feat[ref_mask > 0]
    emb = tgt.mean(0).unsqueeze(0)
    feat = emb / emb.norm(dim=-1, keepdim=True)
    return feat, emb.unsqueeze(0)

def pair_predict(predictor, target_feat, target_emb):
    """调用前 predictor 已 set_image(scene)。返回 (pred_xy or None, score, npix)"""
    test_feat = predictor.features.squeeze()
    C, h, w = test_feat.shape
    test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
    sim = target_feat @ test_feat.reshape(C, h * w)
    sim = sim.reshape(1, 1, h, w)
    sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
    sim = predictor.model.postprocess_masks(sim, input_size=predictor.input_size,
                                            original_size=predictor.original_size).squeeze()
    p_xy, p_lab, n_xy, n_lab = point_selection(sim, topk=1)
    xy = np.concatenate([p_xy, n_xy], axis=0); lab = np.concatenate([p_lab, n_lab], axis=0)
    sn = (sim - sim.mean()) / torch.std(sim)
    sn = F.interpolate(sn.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
    attn = sn.sigmoid_().unsqueeze(0).flatten(3)
    masks, scores, logits, _ = predictor.predict(point_coords=xy, point_labels=lab,
                                                 multimask_output=False, attn_sim=attn,
                                                 target_embedding=target_emb)
    masks, scores, logits, _ = predictor.predict(point_coords=xy, point_labels=lab,
                                                 mask_input=logits[0:1], multimask_output=True)
    best = int(np.argmax(scores))
    y, x = np.nonzero(masks[best])
    if len(x) == 0:
        return None, float(scores[best]), 0
    box = np.array([x.min(), y.min(), x.max(), y.max()])
    masks, scores, logits, _ = predictor.predict(point_coords=xy, point_labels=lab,
                                                 box=box[None, :], mask_input=logits[best:best + 1],
                                                 multimask_output=True)
    best = int(np.argmax(scores))
    y, x = np.nonzero(masks[best])
    if len(x) == 0:
        return None, float(scores[best]), 0
    return (float(x.mean()), float(y.mean())), float(scores[best]), int(len(x))

def load_annot(sj):
    j = json.loads(Path(sj).read_text(encoding="utf-8"))
    tgt, dis = None, []
    for s in j.get("shapes", []):
        p = np.array(s["points"], dtype=np.float32)
        box = [float(p[:, 0].min()), float(p[:, 1].min()), float(p[:, 0].max()), float(p[:, 1].max())]
        if s.get("flags", {}).get("target") is True: tgt = box
        elif s.get("flags", {}).get("target") is False: dis.append(box)
    return tgt, dis, j.get("flags", {}).get("difficulty", "?")

def pairs_for(split, sample=0, seed=42, limit=0, include_all=False, only_inst=""):
    pairs, by_inst = [], {}
    for inst_dir in sorted((Path(DATA) / split).iterdir()):
        if not inst_dir.is_dir(): continue
        if only_inst and inst_dir.name != only_inst: continue
        meta = json.loads((inst_dir / "meta.json").read_text(encoding="utf-8"))
        if meta.get("rgb_usable") is False and not include_all: continue
        tmpls = sorted(int(p.stem) for p in (inst_dir / "rgb" / "template").glob("*.jpg"))
        for sj in sorted((inst_dir / "ir" / "search").glob("*.json")):
            sf = int(sj.stem)
            for t in tmpls:
                if abs(sf - t) >= 30:
                    by_inst.setdefault(inst_dir.name, []).append((inst_dir, meta.get("class", "?"), t, sj))
    for v in by_inst.values(): pairs.extend(v)
    if limit: pairs = pairs[:limit]
    if sample:
        import random as _r
        _r.seed(seed); total = len(pairs); picked = []
        for name, lst in sorted(by_inst.items()):
            k = max(1, round(sample * len(lst) / total))
            picked.extend(_r.sample(lst, min(k, len(lst))))
        pairs = picked
    return pairs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-all-instances", action="store_true", help="不跳过 rgb_usable=false 实例（对齐官方 307,527 口径）")
    ap.add_argument("--only-inst", default="", help="只跑指定实例（补跑用）")
    a = ap.parse_args()
    avail = os.popen("free -m | awk '/^Mem:/{print $7}'").read().strip()
    print("AVAIL_MB", avail, flush=True)
    if int(avail) < 3500:
        print("MEM_GUARD_ABORT", flush=True); return
    pred = build_predictor()
    pairs = pairs_for(a.split, a.sample, a.seed, a.limit, a.include_all_instances, a.only_inst)
    print("PAIRS", len(pairs), flush=True)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try: done.add(json.loads(line)["key"])
            except: pass
    t0 = time.time(); n = 0
    cache = {}
    with open(a.out, "a") as f:
        # 按实例分组：模板 embedding 预计算一次
        groups = {}
        for it in pairs:
            groups.setdefault(it[0], {})[it[2]] = True
        for inst_dir, tmpl_set in groups.items():
            for t in tmpl_set:
                tp = inst_dir / "rgb" / "template" / f"{t:06d}.jpg"
                cache[(inst_dir.name, t)] = ref_embedding(pred, tp)
        # 场景为主循环：每个场景 set_image 一次，然后遍历该实例所有 (tmpl, scene) 对
        by_scene = {}
        for it in pairs:
            by_scene.setdefault((it[0].name, it[3].stem), []).append(it)
        for (iname, sname), items in by_scene.items():
            inst_dir, cls, _t, sj = items[0]
            scene_img = cv2.cvtColor(cv2.imread(str(sj.with_suffix(".jpg"))), cv2.COLOR_BGR2RGB)
            tgt, dis, diff = load_annot(sj)
            if tgt is None: continue
            pred.set_image(scene_img)
            for inst_dir2, cls2, t, sj2 in items:
                key = f"{iname}_{t}_{sname}"
                if key in done: continue
                feat, emb = cache[(iname, t)]
                st = time.time()
                try:
                    xy, score, npix = pair_predict(pred, feat, emb)
                except Exception as e:
                    print("ERR", key, repr(e)[:150], flush=True); continue
                if xy is None:
                    rec = dict(key=key, pck=0, empty=1, score=round(score, 5), pred=None,
                               inst=iname, tmpl=t, scene=int(sname), diff=diff,
                               ms=round((time.time() - st) * 1000, 1))
                else:
                    gx, gy = (tgt[0] + tgt[2]) / 2, (tgt[1] + tgt[3]) / 2
                    r = 0.5 * max(tgt[2] - tgt[0], tgt[3] - tgt[1])
                    pck = int(((xy[0] - gx) ** 2 + (xy[1] - gy) ** 2) ** 0.5 < r)
                    ind = int(any(b[0] <= xy[0] <= b[2] and b[1] <= xy[1] <= b[3] for b in dis))
                    rec = dict(key=key, pck=pck, empty=0, score=round(score, 5),
                               pred=[round(xy[0], 1), round(xy[1], 1)], gt=[round(gx, 1), round(gy, 1)],
                               radius=round(r, 1), in_distractor=ind, npix=npix,
                               inst=iname, tmpl=t, scene=int(sname), diff=diff,
                               ms=round((time.time() - st) * 1000, 1))
                f.write(json.dumps(rec) + "\n"); f.flush(); n += 1
                if n % 200 == 0:
                    el = time.time() - t0
                    print(f"PROGRESS {n} elapsed {el/60:.1f} min vs {n} pairs, {el/n:.2f}s/pair", flush=True)
    print("DONE", n, "in", round(time.time() - t0, 1), "s", flush=True)

if __name__ == "__main__":
    main()

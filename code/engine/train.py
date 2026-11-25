# 训练入口
#
# 用法: python -m engine.train --config configs/resnet_correlation.yaml
import os
import argparse
import json
import random
import signal
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from data.cstloc_dataset import CSTLocDataset
from models.localizer import Localizer
from losses.guided_focal import GuidedFocalMSELoss, AverageMeter
from losses.instance_contrastive import InstanceContrastiveLoss
from losses.dual_template_loss import DualTemplateLoss, build_pairing, load_donor_templates
from losses.distractor_margin import distractor_margin_loss
from engine.evaluate import evaluate, format_results


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class _IdxDataset:
    """给 dataset 样本附加原始索引（对偶模板监督需要 batch 内实例名）。"""
    def __init__(self, ds):
        self.ds = ds

    def __getitem__(self, idx):
        s = self.ds[idx]
        s['_idx'] = idx
        return s

    def __len__(self):
        return len(self.ds)

    @property
    def samples(self):
        return self.ds.samples


def build_loader(cfg, split, mode, shuffle):
    ds = _IdxDataset(CSTLocDataset(
        root=cfg['data_root'], split=split, mode=mode,
        img_size=tuple(cfg['img_size']), patch_size=cfg['patch_size'],
        sigma_factor=cfg['sigma_factor'], n_templates=cfg['n_templates'],
        min_gap_frames=cfg['min_gap_frames'],
        instances=cfg.get('instances'),
    ))
    return DataLoader(ds, batch_size=cfg['batch_size'], shuffle=shuffle,
                      num_workers=cfg['num_workers'], pin_memory=True, drop_last=shuffle)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--data-root', default=None)
    ap.add_argument('--save-dir', default=None)
    ap.add_argument('--seed', type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding='utf-8'))
    if args.data_root:
        cfg['data_root'] = args.data_root
    if args.save_dir:
        cfg['save_dir'] = args.save_dir
    if args.seed is not None:
        cfg['seed'] = args.seed

    # IR 编码器选型解析（缺陷修复 2026-09-06）：config 的 ir_encoder 键为基准，
    # CSTLOC_IR_ENCODER 环境变量为显式覆盖；两者同设且不一致即报错。
    # （j2/J3 两次 UNet 误跑的根因：train.py 曾忽略 yaml 键，env 未设时静默默认 unet。）
    yaml_enc = cfg.get('ir_encoder')
    env_enc = os.environ.get('CSTLOC_IR_ENCODER')
    if yaml_enc and env_enc and yaml_enc != env_enc:
        raise ValueError(f"IR 编码器选型冲突：yaml ir_encoder={yaml_enc} vs "
                         f"CSTLOC_IR_ENCODER={env_enc}——保留其一")
    ir_encoder = env_enc or yaml_enc
    if ir_encoder is None:
        raise ValueError("IR 编码器未指定：请在 config 写 ir_encoder 键或设 "
                         "CSTLOC_IR_ENCODER 环境变量（禁止静默默认 unet）")
    print(f"IR 编码器选型：{ir_encoder}（来源：{'环境变量覆盖' if env_enc else 'config yaml'}）")

    if cfg.get('deterministic'):
        # CuBLAS GEMM 复现所必需（CUDA>=10.2）；须在任何 CUDA 初始化前设置
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    set_seed(cfg.get('seed', 42))
    if cfg.get('deterministic'):
        # deterministic: true → set_seed 已含 cudnn.deterministic=True + 固定种子；这里再关 TF32
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        print(f"deterministic 模式：cudnn.deterministic=True + allow_tf32=False + 固定种子")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    save_dir = Path(cfg['save_dir']); save_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(cfg, cfg['train_split'], 'train', shuffle=True)
    # 评测集：test 模式（同一 split 的 test 语义），单模板确定性取中位
    eval_cfg = dict(cfg, n_templates=cfg['eval_n_templates'])
    eval_loader = build_loader(eval_cfg, cfg['eval_split'], 'test', shuffle=False)
    print(f"train {len(train_loader.dataset)} 样本 | eval {len(eval_loader.dataset)} 样本 | device={device}")

    # 实验 W：corr_readout 键（'mlp' | 缺省='max'）→ env 传给 CorrLayer（须在模型构建前，仿 CSTLOC_IR_ENCODER）
    if cfg.get('corr_readout'):
        os.environ['CSTLOC_CORR_READOUT'] = cfg['corr_readout']
        print(f'corr_readout = {cfg["corr_readout"]}（经 env 传给 CorrLayer）')
    # 实验 W2：corr_token_size 键（缺省 7）→ env 传给 CorrelationModulator（须在模型构建前）
    if cfg.get('corr_token_size'):
        os.environ['CSTLOC_CORR_TOKEN_SIZE'] = str(cfg['corr_token_size'])
        print(f'corr_token_size = {cfg["corr_token_size"]}（经 env 传给 CorrelationModulator）')
    # 任务书 1.9：projection_adapter 键（'s3' | 'orth'）→ env 传给 CorrLayer（须在模型构建前）
    adapter_base = cfg.get('projection_adapter')
    if adapter_base:
        assert adapter_base in ('s3', 'orth'), adapter_base
        os.environ['CSTLOC_PROJ_ADAPTER'] = adapter_base
        print(f'projection_adapter = {adapter_base}（冻结基底 + 适配器；基底权重构建后按臂装载）')
    # 任务书 2.1：corr 旁路键（无适配器时取 corr 图做读出监督）→ env（须在模型构建前）
    corr_bypass = bool(cfg.get('corr_bypass'))
    if corr_bypass:
        os.environ['CSTLOC_CORR_BYPASS'] = '1'
        print('corr_bypass = 1（投影保持普通可训练 1×1，corr 图旁路供读出监督）')
    # 任务书 2.5：CMAR 门控注意力读出（默认关）。cmar:true 时三尺度 CorrLayer 各挂一组
    # 零初始化门控注意力读出（cmar.py；torch.empty+专用 generator 初始化，不耗全局 RNG）。
    cmar = bool(cfg.get('cmar', False))
    cmar_mode = cfg.get('cmar_mode', 'gate')               # 消融格①：replace=去γ替换读出
    cmar_gamma_init = cfg.get('cmar_gamma_init', 'zero')   # 消融格②：random=γ随机初始化
    cmar_direction = cfg.get('cmar_direction', 'spatial')  # 消融格③：token=CMAR-g 字面方向
    if cmar:
        assert cfg['modulator'] == 'correlation', 'CMAR 仅适配 correlation 调制器'
        print(f'CMAR 开启：mode={cmar_mode} gamma_init={cmar_gamma_init} '
              f'direction={cmar_direction}（γ 零初始化 → 训练起点 ≡ 基线，G0 审计）')
    # identity_expert 为 random / linear-random 时由 IdentityExpert 按固定 seed 生成，
    # 不依赖外部文件；其余取值仍支持 CSTLOC_EXPERT_DIR 覆盖历史相对路径。
    _exp = cfg.get('identity_expert')
    if _exp and _exp not in ('null_constant', 'random', 'linear-random'):
        _exp_dir = os.environ.get('CSTLOC_EXPERT_DIR')
        if _exp_dir:
            _cand = Path(_exp_dir) / Path(_exp).name
            if _cand.is_file():
                cfg['identity_expert'] = str(_cand)
                print(f'identity_expert -> {cfg["identity_expert"]}')
    model = Localizer(encoder=cfg['encoder'], modulator=cfg['modulator'],
                      pretrained=cfg['pretrained'],
                      pretrained_path=cfg.get('pretrained_path'),
                      ir_encoder=ir_encoder,
                      use_id=cfg.get('identity_expert') is not None,
                      identity_expert=cfg.get('identity_expert'),
                      mask_null_p=cfg.get('mask_null_p', 0.0),
                      cmar=cmar, cmar_mode=cmar_mode, cmar_gamma_init=cmar_gamma_init,
                      cmar_direction=cmar_direction).to(device)

    # 实验 H：InfMAE 末段解冻（差分 lr）。cfg 键缺省=0=行为不变。
    unfreeze_tail = cfg.get('unfreeze_infmae_tail', 0) or 0
    if unfreeze_tail:
        model.ir_encoder.apply_unfreeze(unfreeze_tail)
        model.ir_encoder.eval()
    total = sum(p.numel() for p in model.parameters()) / 1e6
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    if unfreeze_tail:
        n_tail = sum(p.numel() for p in model.ir_encoder.unfrozen_params) / 1e6
        print(f"模型 {cfg['encoder']}+{cfg['modulator']}：total {total:.2f}M | "
              f"trainable {tr:.3f}M（含 InfMAE 解冻末 {unfreeze_tail} block+norm {n_tail:.2f}M）")
    else:
        print(f"模型 {cfg['encoder']}+{cfg['modulator']}：{total:.2f}M 参数 | trainable {tr:.3f}M")
    if cfg.get('cmar_gamma_freeze'):
        # G0b 审计专用：冻结 γ（保持零初始化）。γ≡0 ⇒ w_q/w_k/w_o 梯度恒为 0 ⇒ Adam 零更新，
        # 全程轨迹应与基线逐位一致。正式训练严禁开启。
        assert cmar, 'cmar_gamma_freeze 仅在 cmar:true 时有意义'
        for _lv in ('corr_x3', 'corr_x4', 'corr_x5'):
            getattr(model.modulator, _lv).cmar.gamma.requires_grad_(False)
        print('[G0b 审计] cmar_gamma_freeze：γ 冻结于 0，w_q/w_k/w_o 梯度应恒为 0')
    dbg_steps = int(cfg.get('debug_step_log', 0) or 0)  # G0b 审计：逐步 loss 落盘后提前退出
    _dbg_i = 1
    if dbg_steps:
        _dbg_file = Path(save_dir) / 'e19c_g0b_steps.jsonl'
        if _dbg_file.exists():
            _dbg_file.unlink()
        print(f'[G0b 审计] debug_step_log={dbg_steps}：逐步 loss → {_dbg_file}，达步数即退出')

    if adapter_base:
        # 1.9 基底装载：corr_x3/4/5 的 proj_ir/proj_tmpl w0 冻结权重 ← S3 训练值 / 随机正交阵
        # （T/Phase0a/0b 同组 seed 42 矩阵）；其余参数保持 A2 配方从头随机初始化。
        # 2.5 环境修复（lead 批准）：a2_random_expert ckpt 本机缺失；orth 臂 w0 实值完全来自
        # seed42 正交阵（s3 装载原仅作形状载体），改惰性装载（非 orth 臂仍需 s3 ckpt）。
        # 本机正交阵与 e116b 三种子存档 w0 逐位一致；vs e19_b max|Δ|≈1.3e-7（QR 环境漂移，
        # e116b 已含同幅先例）→ 报告披露。
        from models.orthogonal import random_orthogonal_matrices
        s3_sd = None
        if adapter_base != 'orth':
            s3_sd = torch.load('checkpoints/a2_random_expert/best_model.pth', map_location='cpu')
        orth = random_orthogonal_matrices() if adapter_base == 'orth' else None
        msd = model.state_dict()
        for lv in (3, 4, 5):
            for p in ('proj_ir', 'proj_tmpl'):
                if orth is not None:
                    w = orth[f'x{lv}:{p}'].view_as(
                        msd[f'modulator.corr_x{lv}.{p}.w0.weight'])
                else:
                    w = s3_sd[f'modulator.corr_x{lv}.{p}.weight'].clone()
                msd[f'modulator.corr_x{lv}.{p}.w0.weight'] = w
        model.load_state_dict(msd)
        print(f'1.9 基底装载：{adapter_base}（corr_x3/4/5 proj_ir/proj_tmpl w0 冻结；'
              f'其余参数 A2 配方从头初始化）')

    # 任务书 2.0：温启动开关。给出 warm_start_from 时整体装载该 checkpoint（真·热启动；
    # 形状/键不匹配项自动跳过——适配器等新增参数保持初始化值）。缺省不装载任何外部权重
    # （= 1.9 A/B 两臂与 2.0 scratch 的实际行为：除冻结基底外全部从头训练）。
    warm_from = cfg.get('warm_start_from')
    if warm_from:
        ws = torch.load(warm_from, map_location='cpu')
        msd = model.state_dict()
        loadable = {k: v for k, v in ws.items() if k in msd and msd[k].shape == v.shape}
        model.load_state_dict({**msd, **loadable})
        print(f'warm_start：自 {warm_from} 装载 {len(loadable)}/{len(ws)} 键'
              f'（跳过 {len(ws) - len(loadable)} 键：形状/键不匹配）')
    else:
        print(f"warm_start：不装载外部权重（warm_start={cfg.get('warm_start', True)}；"
              f'1.9 A/B 与 2.0 scratch 均走此路径——见实验结果20.md 前提核验节）')

    # 任务书 1.9：高斯 CE 读出监督（x4 corr）+ 自蒸馏正则（corr → 冻结基底 corr L2）
    ce_w = cfg.get('corr_ce_weight', 0.0)
    distill_w = cfg.get('corr_distill_weight', 0.0)
    gce = None
    if ce_w > 0 or distill_w > 0:
        assert (adapter_base or corr_bypass), \
            'corr_ce_weight 需 projection_adapter 或 corr_bypass 模式（corr 图旁路）'
        assert (not distill_w) or adapter_base, '自蒸馏（corr_distill_weight）需冻结基底（projection_adapter）'
        from losses.gaussian_ce import GaussianCrossEntropyLoss, build_ce_target  # noqa: E402
        from losses.gaussian_ce import sigma_cosine_anneal  # noqa: E402
        gce = GaussianCrossEntropyLoss(tau=cfg.get('corr_ce_tau', 0.07)).to(device)
        print(f'1.9 读出监督：x4 corr softmax(τ={cfg.get("corr_ce_tau", 0.07)}) CE λ={ce_w} | '
              f'自蒸馏 L2 λ={distill_w} | σ 余弦退火 1.5→0.5（kernel 9，随 epoch）')

    criterion = GuidedFocalMSELoss(alpha=cfg['alpha'], pos_weight=cfg['pos_weight'],
                                   suppress_weight=cfg['suppress_weight'])
    cw = cfg.get('contrast_weight', 0.0)
    contrastive = InstanceContrastiveLoss().to(device) if cw > 0 else None
    dw = cfg.get('dual_template', 0)
    dual = DualTemplateLoss().to(device) if dw > 0 else None
    # 实验 P-a：目标-干扰物 margin 项（分支 A，weight = config distractor_margin）
    dm_w = cfg.get('distractor_margin', 0) or 0
    if dm_w > 0:
        print(f'P-a 目标-干扰物 margin 项启用：weight={dm_w}（仅含干扰物且未遮蔽样本生效）')
    donor_tmpls = None
    if dual is not None:
        # 对偶模板配对表（确定性：同类排序下一位/退化 fallback）+ donor 模板预加载
        inst_names = sorted({p[0].name for p in train_loader.dataset.samples})
        pair_table = build_pairing(inst_names)
        donor_tmpls = load_donor_templates(cfg['data_root'], pair_table)
        diag_dir = Path(cfg['save_dir']).resolve().parent.parent / 'diag_out'
        diag_dir.mkdir(parents=True, exist_ok=True)
        (diag_dir / 'g_pairing.json').write_text(
            json.dumps(pair_table, ensure_ascii=False, indent=2), encoding='utf-8')
        n_fb = sum(1 for v in pair_table.values() if v['fallback'])
        print(f"对偶模板: {len(pair_table)} 实例配对 | fallback={n_fb} | 配对表→diag_out/g_pairing.json")
    if unfreeze_tail:
        # 双参数组：解冻组 lr = 主 lr × backbone_lr_scale（默认 0.1 → 1e-5）；其余可训 lr = 主 lr（1e-4）
        blr = cfg['lr'] * cfg.get('backbone_lr_scale', 0.1)
        tail_ids = {id(p) for p in model.ir_encoder.unfrozen_params}
        g_rest = [p for p in model.parameters() if p.requires_grad and id(p) not in tail_ids]
        g_tail = [p for p in model.parameters() if p.requires_grad and id(p) in tail_ids]
        optimizer = torch.optim.Adam([{'params': g_rest, 'lr': cfg['lr']},
                                      {'params': g_tail, 'lr': blr}])
        print(f"双参数组：解冻组 {len(g_tail)} 张量 lr={blr} | 其余 {len(g_rest)} 张量 lr={cfg['lr']}")
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['epochs'])
                 if cfg.get('lr_schedule') == 'cosine' else None)

    # ---- 实验 K：断点续训 + 自动早停 + SIGTERM/SIGINT（新键缺省关闭，行为逐位不变）----
    def save_last(epoch_done):
        """保存 last.pt：model/optimizer(/scheduler)/epoch/best_pck + torch/numpy/random RNG 全状态。"""
        st = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch_done,
            'best_pck': best_acc,
            'torch_rng': torch.get_rng_state(),
            'cuda_rng': torch.cuda.get_rng_state_all(),
            'numpy_rng': np.random.get_state(),
            'py_rng': random.getstate(),
        }
        if scheduler is not None:
            st['scheduler'] = scheduler.state_dict()
        torch.save(st, save_dir / 'last.pt')

    def restore_rng(st):
        torch.set_rng_state(st['torch_rng'])
        torch.cuda.set_rng_state_all(st['cuda_rng'])
        np.random.set_state(st['numpy_rng'])
        random.setstate(st['py_rng'])

    best_acc = 0.0
    no_improve = 0
    gate_trig = 0
    start_epoch = 1
    resume_path = cfg.get('resume')
    if resume_path:
        rp = Path(resume_path)
        assert rp.exists(), f'resume 路径不存在: {rp}'
        st = torch.load(rp, map_location='cpu')
        model.load_state_dict(st['model'])
        optimizer.load_state_dict(st['optimizer'])
        if scheduler is not None and 'scheduler' in st:
            scheduler.load_state_dict(st['scheduler'])
        best_acc = st['best_pck']
        start_epoch = st['epoch'] + 1
        restore_rng(st)
        print(f'resume: 从 {rp} 恢复（epoch {st["epoch"]} 完成，续从 {start_epoch}）| best_pck={best_acc:.4f}')

    def _on_signal(signum, frame):
        # 收到 SIGTERM/SIGINT：保存 last.pt 再退出（epoch 记当前循环值；eval 边界 kill 为干净断点）
        print(f'[train] 收到信号 {signum}，保存 last.pt 后退出（epoch={epoch}）', flush=True)
        try:
            save_last(epoch)
        except Exception as e:  # noqa: BLE001
            print(f'[train] last.pt 保存失败: {e}', flush=True)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _on_signal)
    signal.signal(signal.SIGINT, _on_signal)

    es_patience = cfg.get('early_stop_patience', 0)
    if es_patience:
        print(f'early_stop_patience={es_patience}（eval 次数；best 连续无改善达该次数且 epoch≥40 停止）')

    for epoch in range(start_epoch, cfg['epochs'] + 1):
        model.train()
        meter = AverageMeter()
        c_meter = AverageMeter()
        d_meter = AverageMeter()
        dn_meter = AverageMeter()
        dm_meter = AverageMeter()
        ce_meter = AverageMeter()
        dist_meter = AverageMeter()
        for batch in train_loader:
            ir = batch['ir'].to(device)
            templates = batch['templates'].to(device)
            heatmap = batch['heatmap'].to(device)
            supp = batch['suppression'].to(device)

            if dual is not None:
                # 对偶模板：IR 特征只过一次（缓存），pos/neg 各一次模板侧前向（共享 IR 特征）
                idxs = batch['_idx'].tolist()
                inst_names = [train_loader.dataset.samples[i][0].name for i in idxs]
                donor_t = torch.stack([donor_tmpls[n] for n in inst_names]).to(device)
                ir_feats = model.ir_encoder(ir)
                mask_null = model.use_id and model.mask_null_p > 0
                out = model(ir, templates, ir_feats=ir_feats, mask_null=mask_null,
                            return_aux=mask_null)
                logits = out if not mask_null else out[0]
                id_mask = out[1]['id_mask'] if mask_null else None
                # 任务书 1.9：读出监督 + 自蒸馏（正样本前向的 x4 corr；dual 负前向会覆盖
                # stash，故必须在负前向之前截获）
                corr_aux = None
                if gce is not None:
                    corr4 = model.modulator.corr_x4._last_corr
                    hm_src = heatmap
                    if mask_null and id_mask is not None and id_mask.any():
                        hm_src = hm_src.clone()
                        hm_src[id_mask] = 0.0
                    t4 = build_ce_target(hm_src, sigma_cosine_anneal(epoch, cfg['epochs']))
                    assert t4.shape[2:] == corr4.shape[2:], \
                        f'CE 目标 {tuple(t4.shape[2:])} vs corr_x4 {tuple(corr4.shape[2:])}（stride 8 假设）'
                    ce = gce(corr4, t4)
                    if distill_w > 0:                       # 任务书 2.1：无冻结锚点时不施自蒸馏
                        corr4b = model.modulator.corr_x4._last_corr_base.detach()
                        dist = (corr4 - corr4b).pow(2).mean()
                    else:
                        dist = torch.zeros((), device=corr4.device)
                    corr_aux = (ce, dist)
                if mask_null and id_mask is not None and id_mask.any():
                    # 遮蔽样本：focal 目标置全零热图（无 id 指引 → 只靠 IR 结构）
                    hm_eff = heatmap.clone()
                    hm_eff[id_mask] = 0.0
                    loss = criterion(logits, hm_eff, supp)
                    # 对偶监督只对未遮蔽样本
                    logits_neg = model(ir, donor_t, ir_feats=ir_feats, mask_null=False)
                    m_loss, n_loss = dual(logits[~id_mask], logits_neg[~id_mask],
                                          heatmap[~id_mask])
                    loss = loss + m_loss + 0.1 * n_loss
                else:
                    loss = criterion(logits, heatmap, supp)
                    logits_neg = model(ir, donor_t, ir_feats=ir_feats, mask_null=False)
                    m_loss, n_loss = dual(logits, logits_neg, heatmap)
                    loss = loss + m_loss + 0.1 * n_loss
                if dm_w > 0:
                    # P-a margin：正确模板前向（logits），仅未遮蔽样本生效（遮蔽样本 id_emb=null，
                    # 其 focal 目标已置零且 logits 非正确模板前向）；不增加前向次数
                    valid_dm = None if id_mask is None else ~id_mask
                    l_dist = distractor_margin_loss(logits, heatmap, supp, valid_dm)
                    loss = loss + dm_w * l_dist
                    dm_meter.update(float(l_dist))
                if corr_aux is not None:
                    ce, dist = corr_aux
                    loss = loss + ce_w * ce + distill_w * dist
                    ce_meter.update(float(ce)); dist_meter.update(float(dist))
                d_meter.update(m_loss.item())
                dn_meter.update(n_loss.item())
            elif contrastive is not None:
                out = model(ir, templates, return_aux=True)
                logits, aux = out
                loss = criterion(logits, heatmap, supp)
                c_loss = contrastive(aux['x5'], aux['proto'], heatmap, supp)
                loss = loss + cw * c_loss
                c_meter.update(c_loss.item())
            else:
                logits = model(ir, templates)
                loss = criterion(logits, heatmap, supp)

            optimizer.zero_grad()
            loss.backward()
            if dbg_steps and cmar:
                # G0b 审计断言：γ 冻结 ⇒ 非 γ CMAR 参数梯度必须逐位为 0；γ 不应收到梯度
                for _lv in ('corr_x3', 'corr_x4', 'corr_x5'):
                    _m = getattr(model.modulator, _lv).cmar
                    assert _m.gamma.grad is None, f'{_lv}.cmar.gamma 竟收到梯度（γ 应被冻结）'
                    for _nm in ('w_q', 'w_k', 'w_o'):
                        _g = getattr(_m, _nm).grad
                        assert _g is not None and float(_g.abs().max()) == 0.0, \
                            f'{_lv}.cmar.{_nm} 梯度非零（γ=0 时应逐位为 0）'
            optimizer.step()
            meter.update(loss.item(), ir.size(0))
            if dbg_steps:
                with open(_dbg_file, 'a') as _f:
                    _f.write(json.dumps({'step': _dbg_i, 'loss': loss.item()}) + '\n')
                _dbg_i += 1
                if _dbg_i > dbg_steps:
                    print(f'[G0b 审计] 已记录 {dbg_steps} 步 loss → {_dbg_file}，退出')
                    return

        log = f"[{epoch:3d}/{cfg['epochs']}] train_loss={meter.avg:.4f}"
        log += f" | lr={optimizer.param_groups[0]['lr']:.2e}"
        if contrastive is not None:
            log += f" | contrast_loss={c_meter.avg:.4f}"
        if dual is not None:
            log += f" | margin_loss={d_meter.avg:.4f} neg_loss={dn_meter.avg:.4f}"
        if dm_w > 0:
            log += f" | dist_margin={dm_meter.avg:.4f}"
        if gce is not None:
            log += f" | ce={ce_meter.avg:.4f} distill={dist_meter.avg:.4f}"
        if cmar:
            # 任务书 2.5：每 epoch 记录 γ 与 attn_readout 统计（含对偶负模板前向，报告披露口径）
            _lvs = ('corr_x3', 'corr_x4', 'corr_x5')
            _gs = [float(getattr(model.modulator, _lv).cmar.gamma.detach()) for _lv in _lvs]
            _sts = [getattr(model.modulator, _lv).cmar.pop_log_stats() for _lv in _lvs]
            _st = ' '.join(
                (f'{_lv}={_s["mean"]:.4f}[{_s["min"]:.4f},{_s["max"]:.4f}]'
                 if _s else f'{_lv}=无') for _lv, _s in zip(_lvs, _sts))
            log += f' | cmar_γ=[{_gs[0]:.5f},{_gs[1]:.5f},{_gs[2]:.5f}] attn_mean[min,max] {_st}'
        if epoch % cfg['eval_every'] == 0 or epoch == cfg['epochs']:
            if cfg.get('deterministic'):
                print(f'[eval] epoch {epoch} 评测开始', flush=True)   # 仅 deterministic 模式（1.2a 验证用）
            res = evaluate(model, eval_loader, device)
            acc = res['overall'][0]
            log += f" | eval_PCK={acc:.4f}"
            improved = acc > best_acc
            if improved:
                best_acc = acc
                torch.save(model.state_dict(), save_dir / 'best_model.pth')
                log += " *"
            save_last(epoch)               # 每次 eval 后保存 last.pt（含 RNG 全状态）
            no_improve = 0 if improved else no_improve + 1
            print(log)
            print(format_results(res))
            if es_patience and epoch >= 40 and no_improve >= es_patience:
                print(f'早停：best 连续 {no_improve} 次 eval 无改善（epoch≥40），停止于 epoch {epoch}')
                break
            if cfg.get('collapse_gate'):
                # 任务书 1.9 训练期塌缩门禁（2.1 可配置只记录不早停）：每 eval 跑探针（489 trials）
                gate_stop = bool(cfg.get('collapse_gate_early_stop', True))
                from tools.e19_collapse_probe import probe_unseen_median_x4  # noqa: E402
                if 'probe_sample' not in locals():
                    from tools.t_collapse_diag import build_sample, CACHE as T_CACHE  # noqa: E402
                    probe_sample = [s for s in build_sample()[0]]
                    probe_cache = torch.load(T_CACHE, map_location='cpu')
                um = probe_unseen_median_x4(model, probe_sample, probe_cache)
                base = cfg['collapse_gate_baseline']
                trig = um is not None and um < base - 0.05
                print(f"[gate] ep{epoch} 未见池 median r_x4 = {um}（基底 {base}，"
                      f"阈 {base - 0.05:.3f}）{'触发' if trig else 'OK'}", flush=True)
                if trig:
                    gate_trig += 1
                    if gate_trig >= 2 and gate_stop:
                        print('塌缩门禁：连续两次触发 → 早停上报（适配器未挡住塌缩，进负结果分析）')
                        break
                else:
                    gate_trig = 0
        else:
            print(log)
        if scheduler is not None:
            scheduler.step()

    print(f"\n最佳 eval PCK@0.5 = {best_acc:.4f}")


if __name__ == '__main__':
    main()

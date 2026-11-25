# PerSAM 对照臂

状态：v3 test 全量完成（327,772 配对 / 24 个有效实例）。

设计：参考 = RGB 模板（中心 80% 作为 mask）；目标 = IR 场景；training-free PerSAM 得到最终
mask 质心作为预测点；mask 为空记 miss，不做回退。

## 资产与运行

| 项 | 值 |
|---|---|
| 上游代码 | references/repos/Personalize-SAM（官方 tarball，MIT） |
| SAM ViT-H | references/weights/sam_vit_h_4b8939.pth（2.42GB，不入库） |
| 环境 | conda countgd（torch 2.2.1+cu121，含 segment-anything） |
| 数据 | CSTLOC_DATA_ROOT，默认 /home/user/Code/Datasets/CSTLoc |
| 运行 | python run_cstloc.py --split test --out results_test.jsonl（增量可续跑） |

可用 CSTLOC_PERSAM_REPO / CSTLOC_SAM_CKPT 覆盖上游代码与 SAM 权重路径。

## v3 test 结果

| 指标 | PerSAM |
|---|---:|
| pair-macro PCK@0.5·max | 0.0440 |
| trial-micro overall | 0.0510 |
| easy / hard | 0.0640 / 0.0354 |
| seen / unseen | 0.0414 / 0.0639 |
| PCK@0.5·min | 0.0206 |
| AUC（τ 0.1–1.0） | 0.0540 |
| 单帧时延 | 339.0 ms（n=2,083；334 帧空 mask） |

失败分析：非空预测中仅少数落入同类干扰物框，大量落在画面其它区域；跨模态参考外观无法在
红外场景中保持身份一致，失败以背景/漏检与实例混淆为主。

# DINOv2 检索对照臂

冻结 DINOv2 ViT-L（timm vit_large_patch14_dinov2.lvd142m）做模板—场景逐位置余弦检索：
IR 场景 letterbox 到 384×640，零填充到 392×644 以适配 patch14；取最后一个 block 的特征图，
裁回有效区 26×45；模板 resize 224×224 后取 CLS+patch 的 257-token GAP，L2 归一化；
两者逐位置余弦相关，argmax 为预测，另落 top-32 峰（NMS 32px）。

## 环境与运行

~~~bash
export CSTLOC_DATA_ROOT=/path/to/CSTLoc
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
python baselines/dinov2_retrieval/run_cstloc.py --tag test
python baselines/dinov2_retrieval/measure_latency.py
~~~

依赖 timm 的 DINOv2 权重（默认走本机 HF 缓存；首次运行需联网或预置权重）。

## v3 test 结果

| 指标 | DINOv2 检索 |
|---|---:|
| pair-macro PCK@0.5·max | 0.2071 |
| trial-micro overall | 0.2302 |
| easy / hard | 0.3190 / 0.1783 |
| seen / unseen | 0.2015 / 0.2504 |
| PCK@0.5·min | 0.1053 |
| AUC（τ 0.1–1.0） | 0.1970 |
| 单帧时延 | 58.9 ms（RTX 4090，batch=1） |

## 产物

| 文件 | 内容 |
|---|---|
| results.jsonl | 逐 trial 预测点（原图 1920×1080） |
| peaks.jsonl | top-32 峰（原图坐标 + 余弦分） |
| metrics.json | v3 test 聚合指标（随包） |

results.jsonl / peaks.jsonl 体积大，不随包；按上面的命令重建。模板描述子为 257-token GAP，
检索粒度为 patch 级（letterbox 步长 14px，原图约 42px）；该粒度对 test 小目标构成范式级上限。

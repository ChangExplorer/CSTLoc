# baselines/ — 五个对照方法

最终主表只保留五个对照，均按各自官方推荐方式适配到 CSTLoc：

| 目录 | 方法 | v3 test overall | hard |
|---|---|---|---|
| `PerSAM/` | 示例条件化分割（training-free PerSAM） | 0.0440 | 0.0354 |
| `ncc/` | 归一化互相关模板匹配（CPU） | 0.0633 | 0.0437 |
| `xmodal_matcher/` | 跨模态稠密匹配（RoMa 官方权重） | 0.1745 | 0.1377 |
| `dinov2_retrieval/` | 冻结 DINOv2 特征检索 + 滑窗 | 0.2071 | 0.1783 |
| `owv_detector/` | 开放词汇检测（Grounding DINO） | 0.2116 | 0.1902 |

数字为 v3 test 327,772 配对上的 pair-macro PCK@0.5·max。每个子目录随包附带 metrics.json（同一 v3 口径）；results.jsonl / peaks.jsonl 体积大，不随包，按各子目录命令可重建。

## 共同契约

每个方法只输出 `results.jsonl`（必要时加 `peaks.jsonl`），格式见 `evaluation/README.md`：
坐标为场景图原始 1920×1080 像素，只存预测，不存 GT/难度/类别。聚合统一走
`evaluation/aggregate.py`，不另写评测逻辑。

## 运行

每个子目录一个自包含入口 `run_cstloc.py`，例如：

```bash
python baselines/ncc/run_cstloc.py --split test --out results.jsonl
python baselines/dinov2_retrieval/run_cstloc.py --split test
python baselines/xmodal_matcher/run_cstloc.py --split test
python baselines/owv_detector/run_cstloc.py --split test
python baselines/PerSAM/run_cstloc.py --split test --out results.jsonl
```

依赖、权重来源、输出文件与逐方法细节见各子目录 `README.md`。

## 大文件

`baselines/*/*.jsonl`、`*.pt`、`*.pth`、分片与特征缓存一律不入 Git；上游仓库与权重
按各子目录 README 的说明自行获取。

# 对照方法证据说明

最终主表使用五个对照，v3 test（327,772 配对，pair-macro PCK@0.5·max）：

| 方法 | overall | easy | hard | 来源 |
|---|---:|---:|---:|---|
| PerSAM | 0.0440 | 0.0640 | 0.0354 | 官方 training-free 流程 |
| NCC | 0.0633 | 0.0970 | 0.0437 | 灰度多尺度模板匹配 |
| RoMa | 0.1745 | 0.2558 | 0.1377 | 官方权重，稠密匹配 + 置信度质心 |
| DINOv2 检索 | 0.2071 | 0.3190 | 0.1783 | 冻结 DINOv2 全局描述子 + 滑窗 |
| Grounding DINO | 0.2116 | 0.3053 | 0.1902 | 类别文本提示 + 检测框中心 |

- 逐 trial 底稿与 peaks 在 `evaluation/cache/v3_comparison/`（本地缓存，不入 Git）。
- 每个方法的适配方式与局限见对应子目录 `README.md`；统一契约见 `evaluation/README.md`。
- 论文中的对照数字来自 v3 全量 test，不使用 v2 历史数字。
- 上游代码/权重获取方式见各子目录 README；权重不入库。

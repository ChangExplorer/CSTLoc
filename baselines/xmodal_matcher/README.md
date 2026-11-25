# 跨模态匹配器对照臂（RoMa）

- 官方仓库：references/repos/RoMa（Parskatt/RoMa tarball）
- 官方权重：references/weights/roma_outdoor.pth（另存 hub 缓存）
- 配置：roma_outdoor(coarse_res=560, upsample_res=(864,1152), use_custom_corr=False)，
  使用官方支持的回退路径；本机未编译 fused-local-corr。
- 预测：top-200 置信匹配点的置信度加权质心（场景侧），坐标映射回原始 1920×1080；
  peaks = top-32 匹配置信点。
- 运行：
  ~~~bash
  export CSTLOC_DATA_ROOT=/path/to/CSTLoc
  python baselines/xmodal_matcher/run_cstloc.py --split test
  ~~~

## v3 test 结果

| 指标 | RoMa |
|---|---:|
| pair-macro PCK@0.5·max | 0.1745 |
| trial-micro overall | 0.1798 |
| easy / hard | 0.2558 / 0.1377 |
| seen / unseen | 0.1768 / 0.1563 |
| PCK@0.5·min | 0.1146 |
| AUC（τ 0.1–1.0） | 0.1730 |
| 单帧时延 | 749.3 ms（RTX 4090，batch=1） |

RoMa 能建立像素级对应，但输出的是几何匹配而非实例身份；失败以背景/漏检与同类实例混淆为主。

# 开放词汇检测器对照臂（Grounding DINO）

- 官方仓库：references/repos/GroundingDINO（IDEA-Research tarball）
- 官方权重：references/weights/groundingdino_swint_ogc.pth（release v0.1.0-alpha）
- 环境补丁（入库文件即已打补丁版本）：
  1. CUDA 扩展：TORCH_CUDA_ARCH_LIST="8.9" python setup.py build_ext --inplace
  2. bertwarper.py 两处新版 transformers 兼容
  3. 依赖：addict / yapf / pycocotools；HF_ENDPOINT=https://hf-mirror.com
- 运行：
  ~~~bash
  export CSTLOC_DATA_ROOT=/path/to/CSTLoc
  HF_ENDPOINT=https://hf-mirror.com python baselines/owv_detector/run_cstloc.py
  ~~~
- 口径：每（实例, 帧）以实例类别名为文本提示做一次前向，取最高分框中心（原图坐标）；
  检测器不使用模板，故同实例所有模板共享同一预测。top-1 无阈值。

## v3 test 结果

| 指标 | Grounding DINO |
|---|---:|
| pair-macro PCK@0.5·max | 0.2116 |
| trial-micro overall | 0.2690 |
| easy / hard | 0.3053 / 0.1902 |
| seen / unseen | 0.2277 / 0.0876 |
| PCK@0.5·min | 0.1740 |
| AUC（τ 0.1–1.0） | 0.2225 |
| 单帧时延 | 114.5 ms（RTX 4090，batch=1） |

该臂能给出大致区域，但没有模板身份约束；失败以背景/漏检和同类实例混淆为主。

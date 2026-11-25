# 跨模态匹配器臂（RoMa，任务书 2.4 臂 A）

- 官方仓库：references/repos/RoMa（Parskatt/RoMa tarball）；官方权重：references/weights/roma_outdoor.pth（另存 hub 缓存）。
- 配置：roma_outdoor(coarse_res=560, upsample_res=(864,1152), **use_custom_corr=False**)
  ——RoMa 官方支持的回退路径（非 Linux 平台同款）；fused-local-corr 本机源码编译未竞，
  故取该回退（与官方默认输出一致时会在报告注明差异为 0 的复现校验）。
- 坐标：RoMa 输出为归一化 [-1,1] 坐标（场景侧），映射回原始 1920×1080；同图自检质心
  [970.8, 531.8] ≈ [960, 540]（容差 2% 内）通过。
- 预测：top-200 置信匹配点的置信度加权质心（场景侧）；peaks = top-32 匹配置信点。
- 运行：双 GPU 分实例并行（GPU0 后 12 实例 / GPU1 前 12 实例），断点 roma_progress.json。

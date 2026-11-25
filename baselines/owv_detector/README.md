# 开放词汇检测器臂（GroundingDINO，任务书 2.4 臂 B）

- 官方仓库：references/repos/GroundingDINO（tarball 解包，IDEA-Research）
- 官方权重：references/weights/groundingdino_swint_ogc.pth（release v0.1.0-alpha）
- 环境补丁（已入库文件即已打补丁版本）：
  ① CUDA 扩展编译：`TORCH_CUDA_ARCH_LIST="8.9" python setup.py build_ext --inplace`（产物已生成）；
  ② bertwarper.py 两处新版 transformers 兼容（get_head_mask 移除回退 / get_extended_attention_mask 两参化）；
  ③ 依赖：addict / yapf / pycocotools；`HF_ENDPOINT=https://hf-mirror.com`（bert-base-uncased 需要）。
- 运行：`HF_ENDPOINT=https://hf-mirror.com CUDA_VISIBLE_DEVICES=1 python baselines/owv_detector/run_cstloc.py [--instances car_001,bus_025]`
- 口径：每（实例,帧）以实例类别名为文本提示一次前向，取最高分框中心（原图坐标）；
  检测器不使用模板，故同实例所有模板共享同一预测。top-1 无阈值（恒有输出）。
- 实测：test 对宏 0.2251（easy 0.3366 / hard 0.1926；双跑 11,580 行 sha256 逐位一致）。

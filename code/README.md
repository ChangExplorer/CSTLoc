# code/ — 最终系统实现

最终系统是串行两段：**冻结双骨干定位器**（DINOv2 模板 + InfMAE 红外 → 三尺度 CorrLayer → UNet 热图）
和 **TCIA 身份适配器**（top-20 候选 → 双角色 LoRA 身份嵌入 → 余弦 argmax）。定位器可训练 6.44M，
TCIA 3.02M，全系统约 9.46M。

## 模块地图

| 路径 | 作用 |
|---|---|
| `configs/final/e116b_s42.yaml` | 定位器最终配置（seed 42，v3 数据） |
| `configs/final/tcia_v2_s2024.yaml` | TCIA 最终配方（seed 2024；超参实际在训练脚本内） |
| `models/localizer.py` | 定位器：DINOv2/InfMAE 双分支、CorrLayer、UNet 热图头 |
| `models/encoders/` | DINOv2 与 InfMAE 冻结编码器 |
| `models/modulators/` | CorrLayer 相关调制与消融用的 simple fusion |
| `models/tcia_v2.py` / `models/tcia.py` | TCIA 双角色 LoRA 验证器 |
| `data/cstloc_dataset.py` | CSTLoc 数据集读取与模板/场景采样 |
| `data/heatmap.py` | GT 各向异性高斯热图与干扰物抑制掩码 |
| `losses/guided_focal.py` | 定位器主损失：前景加权 focal MSE + 干扰物抑制 |
| `losses/dual_template_loss.py` | 对偶模板 margin / negative 损失 |
| `losses/gaussian_ce.py` | x4 相关图交叉熵与自蒸馏 |
| `engine/train.py` | 定位器训练入口（支持 --data-root/--save-dir/--seed） |
| `tools/tcia_v2_train.py` | TCIA 训练入口 |
| `tools/eval_ablation_tcia.py` | 最终 test 评测入口，输出 summary 与逐 trial 结果 |
| `tools/tcia_data.py` / `tools/tcia_v2_data.py` | 数据路径、定位器 checkpoint 与候选读取 |
| `tools/tcia_v2_eval.py` | TCIA 离线评测 |
| `smoke_test.py` | 模型前反向 + 数据集读取冒烟测试（不依赖权重） |

历史实验脚本与历史配置仅供追溯，不参与最终复现；发布包只保留最终闭包。

`code/diag_out/`、`code/checkpoints/`、`code/logs/` 不入 Git。

## 快速检查

```bash
cd code
python smoke_test.py
```

数据集根由 `CSTLOC_DATA_ROOT` 指定，默认 `/home/user/Code/Datasets/CSTLoc`。
完整训练与评测命令见根目录 `scripts/` 与 `releases/CSTLoc/README.md`。

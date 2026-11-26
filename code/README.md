# code/ — system implementation

The released system is a serial two-stage pipeline: a **frozen dual-backbone locator**
(DINOv2 template tower + InfMAE IR tower → three-scale CorrLayer → UNet heatmap head)
and the **TCIA identity adapter** (top-20 candidates → dual-role LoRA identity
embeddings → cosine argmax). Trainable parameters: 6.44M (locator) + 3.02M (TCIA)
≈ 9.46M total.

## Module map

| Path | Role |
|---|---|
| `configs/final/locator.yaml` | locator training config (v3 data) |
| `configs/final/tcia.yaml` | TCIA recipe reference (hyperparameters live in `tools/tcia_train.py`) |
| `models/localizer.py` | locator: dual encoders, CorrLayer, UNet heatmap head |
| `models/encoders/` | frozen DINOv2 / InfMAE encoders (+ InfMAE vendor code) |
| `models/modulators/` | CorrLayer and the simple-fusion ablation |
| `models/tcia.py` | TCIA / TCIAv2: dual-role LoRA verifier |
| `models/identity_expert.py` | code-generated random identity expert head |
| `data/cstloc_dataset.py` | dataset reading, template/scene sampling |
| `data/heatmap.py` | anisotropic Gaussian GT heatmaps, distractor suppression masks |
| `losses/` | guided focal MSE, dual-template margin, correlation CE/distillation |
| `engine/train.py` | locator training entry (`--config/--data-root/--save-dir/--seed`) |
| `tools/paths.py` | dataset/weights path resolution (env-driven, no machine defaults) |
| `tools/image_ops.py` | NMS peaks, 224 crops, normalization, PCK |
| `tools/dataset_io.py` | frames/templates/protocol enumeration, donors |
| `tools/locator.py` | locator construction, cached forward, peak+crop extraction |
| `tools/template_cache.py` | DINOv2 template-feature cache (built from the dataset on demand) |
| `tools/tcia_eval.py` | TCIA val/test evaluation (`run_split`) |
| `tools/tcia_train.py` | TCIA training entry |
| `tools/build_tcia_candidates.py` | rebuild TCIA training candidates from a locator |
| `tools/evaluate_tcia.py` | final test evaluation entry (called by `scripts/05_eval_test.sh`) |
| `smoke_test.py` | model forward/backward + dataset smoke test (no checkpoints needed) |

## Quick check

```bash
cd code
python smoke_test.py
```

The dataset root comes from `CSTLOC_DATA_ROOT` (required, no default).
Full training/evaluation commands live in `scripts/`; see the top-level README.

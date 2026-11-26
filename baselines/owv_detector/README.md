# Open-vocabulary detector baseline (Grounding DINO)

Open-vocabulary detection arm: per (instance, scene frame), run Grounding DINO once
with the instance class name as the text prompt and take the center of the
highest-scoring box (original-image coordinates). The detector does **not** use the
template, so all templates of an instance share the same prediction.

## Setup

1. **Clone the official repo** into `references/repos/GroundingDINO`
   (`references/` is git-ignored, create it yourself):

   ```bash
   mkdir -p references/repos references/weights
   git clone https://github.com/IDEA-Research/GroundingDINO references/repos/GroundingDINO
   ```

2. **Apply the compatibility patch** (required; upstream was written for
   transformers 4.x, we run transformers 5.8). From the GroundingDINO repo root:

   ```bash
   cd references/repos/GroundingDINO
   patch -p1 < ../../../baselines/owv_detector/groundingdino_compat.patch
   ```

   The patch touches only `groundingdino/models/GroundingDINO/bertwarper.py`
   (two hunks, both backward-compatible with old transformers):

   - `BertModel.get_head_mask` was removed in newer transformers → fall back to a
     "no head mask" lambda via `getattr`;
   - `get_extended_attention_mask` now expects `dtype` (not `device`) as its third
     argument → call it with two arguments.

   Applying it by hand is equally fine (4 changed lines; see the patch file).

3. **Build the CUDA extension** (multi-scale deformable attention):

   ```bash
   TORCH_CUDA_ARCH_LIST="8.9" python setup.py build_ext --inplace
   ```

   `8.9` covers RTX 40xx (Ada). Adjust to your GPU (`8.6` for RTX 30xx, `8.0` for
   A100, `9.0` for H100) or drop the variable to auto-detect.

4. **Download the official weights** (`groundingdino_swint_ogc.pth`,
   [release v0.1.0-alpha](https://github.com/IDEA-Research/GroundingDINO/releases/tag/v0.1.0-alpha))
   into `references/weights/groundingdino_swint_ogc.pth`.

5. **Python deps**: `pip install addict yapf pycocotools`
   (plus the repo's own `requirements.txt`; torch/torchvision/transformers already
   installed for the main system).

## Run

```bash
export CSTLOC_DATA_ROOT=/path/to/CSTLoc
python baselines/owv_detector/run_cstloc.py
```

Note: `run_cstloc.py` defaults `HF_ENDPOINT=https://hf-mirror.com` (a mirror useful
where huggingface.co is unreachable); export your own `HF_ENDPOINT` to override.
Outputs `results.jsonl` + `peaks.jsonl` (top-32 detections) in this directory;
instance-level resume via `owv_progress.json`. Aggregate with
`evaluation/aggregate.py` under the same protocol as the main system.

## v3 test results

| Metric | Grounding DINO |
|---|---:|
| pair-macro PCK@0.5·max | 0.2116 |
| trial-micro overall | 0.2690 |
| easy / hard | 0.3053 / 0.1902 |
| seen / unseen | 0.2277 / 0.0876 |
| PCK@0.5·min | 0.1740 |
| AUC (τ 0.1–1.0) | 0.2225 |
| Latency per frame | 114.5 ms (RTX 4090, batch=1) |

The arm finds the rough region but has no template-identity constraint; failures
are dominated by background/missed detections and same-class instance confusion.

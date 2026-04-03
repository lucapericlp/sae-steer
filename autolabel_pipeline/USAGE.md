# SAE Autolabelling Usage

This document describes how to reproduce the SAE feature autolabelling workflow in this repo.

The pipeline is staged to fit on a single NVIDIA A10G:

1. Generate 3 feature-steered SDXL-Turbo images per SAE feature.
2. Unload SDXL.
3. Load `Qwen2.5-VL-7B-Instruct` and label each 3-image feature triplet.

This avoids keeping both large models in VRAM at the same time.

## Environment

Use `uv` and the local `uv venv`.

```bash
cd /home/ubuntu/Laboratory/sae-steer
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
uv pip install --python .venv/bin/python diffusers transformers accelerate safetensors huggingface_hub scipy pandas bitsandbytes sentencepiece protobuf
uv pip install --python .venv/bin/python umap-learn scikit-learn clusterfun
```

Sanity-check CUDA:

```bash
./.venv/bin/python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
PY
```

## Core Scripts

- `autolabel_pipeline/autolabel.py`
  - low-level staged pipeline
  - subcommands: `generate`, `label`, `run`
- `autolabel_pipeline/run_full_labels.py`
  - resumable chunked full-run orchestrator
- `autolabel_pipeline/render_umap_clusterfun.py`
  - render a UMAP scatter for labelled features using `clusterfun`

## Recommended A10G Settings

These settings were benchmarked on this machine:

- generation batch size: `160`
- label batch size: `32`
- CPU workers: `8`
- seeds: `11 37 89`
- steering scale: `120`

Observed ballpark usage:

- SDXL generation: about `20.2 GiB`
- Qwen-VL labelling: about `19.1 GiB`

## Single Run

Run a small staged batch:

```bash
./.venv/bin/python autolabel_pipeline/autolabel.py run \
  --output-dir autolabel_pipeline/runs/test_run \
  --feature-ids 770 4071 4443 6214 7446 \
  --seeds 11 37 89 \
  --steering-scale 120 \
  --generate-batch-size 160 \
  --label-batch-size 32 \
  --cpu-workers 8
```

Outputs:

- `images/`
- `generation_manifest.json`
- `labels.json`
- `labels.csv`
- `gpu_profile.jsonl`
- `stage_summary.jsonl`

## 1024-Feature Benchmark

Example:

```bash
FEATURES=$(seq -s ' ' 0 1023)
./.venv/bin/python autolabel_pipeline/autolabel.py run \
  --output-dir autolabel_pipeline/runs/benchmark_1024_bs160_32 \
  --feature-ids $FEATURES \
  --seeds 11 37 89 \
  --steering-scale 120 \
  --generate-batch-size 160 \
  --label-batch-size 32 \
  --cpu-workers 8
```

## Full 10K Run

The intended production path is to seed the full run from an existing validated benchmark, then process the remaining features in chunks.

Example:

```bash
./.venv/bin/python autolabel_pipeline/run_full_labels.py \
  --source-run autolabel_pipeline/runs/benchmark_1024_bs160_32 \
  --final-run autolabel_pipeline/runs/full_10k_bs160_32 \
  --chunk-size 1024 \
  --generate-batch-size 160 \
  --label-batch-size 32
```

What this does:

1. Copies the benchmarked `0..1023` results into the final run directory.
2. Processes the remaining features in `1024`-feature chunks.
3. Merges each completed chunk into the final run.

Important:

- A chunk is only considered fully complete once `labels.json` exists for that chunk and its outputs have been merged.
- If a chunk has complete generation artifacts but is missing labels, the orchestrator should resume from `label`.
- If you intentionally discard a partial chunk, delete that chunk directory only.

## Monitoring

GPU:

```bash
nvidia-smi
```

Track the current chunk:

```bash
tail -n 20 autolabel_pipeline/runs/full_10k_bs160_32/chunk_runs/chunk_001_1024_2047/gpu_profile.jsonl
```

Count merged labels:

```bash
./.venv/bin/python - <<'PY'
import json
from pathlib import Path
run = Path("autolabel_pipeline/runs/full_10k_bs160_32")
print(len(json.loads((run / "labels.json").read_text())))
PY
```

Inspect live processes:

```bash
ps -eo pid,pcpu,pmem,etime,stat,cmd | rg 'autolabel_pipeline/(run_full_labels|autolabel.py (run|generate|label))'
```

## UMAP Review with Clusterfun

Render a UMAP over an existing labelled subset:

```bash
./.venv/bin/python autolabel_pipeline/render_umap_clusterfun.py \
  --run-dir autolabel_pipeline/runs/benchmark_1024_bs160_32 \
  --output-dir autolabel_pipeline/plots/benchmark_1024_umap
```

Serve the generated `clusterfun` view:

```bash
CLUSTERFUN_PORT=8010 ./.venv/bin/clusterfun <clusterfun-uuid>
```

The script writes `plot_metadata.json` containing the generated UUID and serve command.

## Output Layout

Typical final-run layout:

```text
autolabel_pipeline/runs/full_10k_bs160_32/
  images/
  labels.json
  labels.csv
  generation_manifest.json
  gpu_profile.jsonl
  stage_summary.jsonl
  chunk_runs/
    chunk_001_1024_2047/
    chunk_002_2048_3071/
    ...
```

## Recovery Notes

If the machine stops mid-run:

1. Do not delete the whole final run.
2. Inspect the current chunk directory.
3. If the chunk has:
   - no useful files: rerun the orchestrator
   - complete images and `generation_manifest.json` but no `labels.json`: resume from `label`
   - complete `labels.json`: merge that chunk into the final run or rerun the orchestrator if merge logic already handles it
4. Only delete the specific broken chunk directory if you want to restart that chunk from scratch.

If you want to reset back to only the first 1024 validated features:

```bash
rm -rf autolabel_pipeline/runs/full_10k_bs160_32/chunk_runs/chunk_001_1024_2047
```

Do not delete:

- `autolabel_pipeline/runs/full_10k_bs160_32/labels.json`
- `autolabel_pipeline/runs/full_10k_bs160_32/generation_manifest.json`
- `autolabel_pipeline/runs/full_10k_bs160_32/images/feature_0` through `feature_1023`

## Known Caveats

- `transformers` may print a warning about generation flags such as `temperature`; this did not block successful runs here.
- `clusterfun` shows hovered media in a right-hand preview panel rather than a tiny inline tooltip.
- If a resumed orchestration behaves unexpectedly, inspect the chunk directory before restarting to avoid redundant recomputation.

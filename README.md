# benchmark-vlm-cpu

A benchmarking suite for small Vision-Language Models (VLMs) running on CPU.

Evaluates models across three tasks — image captioning, visual question answering, and structured output extraction — and reports latency, peak memory, and task-specific quality metrics.

## Supported Models

| Short name | Default HF model |
|---|---|
| `qwen3-vl` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| `qwen2.5-omni` | `Qwen/Qwen2.5-Omni-3B` |
| `paligemma2` | `google/paligemma2-3b-pt-224` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run with the default config (editable in configs/default.yaml)
vlm-bench run --config configs/default.yaml

# Run a specific model on specific tasks
vlm-bench run --model qwen3-vl --task captioning --task vqa

# List all registered models and tasks
vlm-bench list
```

Results are saved to `results/results_<timestamp>.json` and `.csv`.

## Data Formats

Place task data under `data/`. Two formats are supported:

**JSON manifest** — `data/<task_name>.json`:
```json
[{"image": "images/dog.jpg", "prompt": "Describe this image.", "reference": "A golden dog..."}]
```

**HuggingFace dataset** — directory at `data/<task_name>/` saved via `dataset.save_to_disk()`. Expected columns: `images`, `messages`, `only_images`.

Generate a manifest from a folder of images:
```bash
python scripts/make_sample_manifest.py data/images \
    --task captioning --prompt "Describe this image." --out data/captioning.json
```

## Tasks and Metrics

| Task | Metrics |
|---|---|
| `captioning` | BLEU-4, ROUGE-L |
| `vqa` | Exact match, Token F1 |
| `structured_output` | Valid JSON rate, Field match ratio |

## Configuration

`configs/default.yaml` controls which models and tasks to run, token limits, and optional `torch.compile` acceleration:

```yaml
models:
  - qwen3-vl
tasks:
  - captioning
  - vqa
  - structured_output
max_new_tokens: 256
compile: false          # set true for ~10-30% speedup (requires C++ compiler)
compile_backend: inductor
```

## Tests

```bash
pytest tests/
```

Tests cover metrics, task prompt building/scoring, and registry integrity — no model weights required.

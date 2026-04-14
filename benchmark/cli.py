"""CLI entry point:  vlm-bench run / vlm-bench list"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer
import yaml
from PIL import Image

from benchmark.models.registry import MODEL_REGISTRY
from benchmark.tasks.registry import TASK_REGISTRY
from benchmark.tasks.base import TaskSample

app = typer.Typer(help="VLM CPU Benchmark Suite", add_completion=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_samples(data_dir: Path, task_name: str, hf_split: str = "test") -> list[TaskSample]:
    """Load samples for a task from either a HF dataset directory or a JSON manifest.

    Resolution order:
    1. ``<data_dir>/<task_name>/dataset_dict.json`` — HuggingFace dataset
       (saved with ``dataset.save_to_disk()``).  ``hf_split`` selects the split.
    2. ``<data_dir>/<task_name>.json`` — plain JSON manifest.
    3. Fallback: a single grey dummy image (warns on stderr).
    """
    # --- HuggingFace dataset ---
    hf_path = data_dir / task_name
    if (hf_path / "dataset_dict.json").exists():
        from benchmark.loaders import load_hf_dataset
        typer.echo(f"[info] Loading HF dataset from {hf_path} (split={hf_split!r})", err=True)
        return load_hf_dataset(hf_path, split=hf_split)

    # --- JSON manifest ---
    manifest = data_dir / f"{task_name}.json"
    if not manifest.exists():
        typer.echo(
            f"[warn] No sample manifest found at {manifest}. "
            "Running with a single dummy sample.",
            err=True,
        )
        dummy_img = Image.new("RGB", (224, 224), color=(128, 128, 128))
        return [TaskSample(image=dummy_img, prompt="Describe this image.", reference=None)]

    entries = json.loads(manifest.read_text())
    samples: list[TaskSample] = []
    for e in entries:
        img_path = Path(e["image"])
        if not img_path.is_absolute():
            img_path = data_dir / img_path
        image = Image.open(img_path).convert("RGB")
        samples.append(
            TaskSample(
                image=image,
                prompt=e.get("prompt", ""),
                reference=e.get("reference"),
                metadata=e.get("metadata", {}),
            )
        )
    return samples


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def run(
    models: Annotated[
        list[str],
        typer.Option("--model", "-m", help="Model short name (repeatable)."),
    ] = list(MODEL_REGISTRY.keys()),
    tasks: Annotated[
        list[str],
        typer.Option("--task", "-t", help="Task name (repeatable)."),
    ] = list(TASK_REGISTRY.keys()),
    data_dir: Annotated[
        Path, typer.Option("--data-dir", help="Directory with sample manifests.")
    ] = Path("data"),
    results_dir: Annotated[
        Path, typer.Option("--results-dir", help="Directory to save results.")
    ] = Path("results"),
    max_new_tokens: Annotated[
        int, typer.Option("--max-tokens", help="Max tokens to generate per sample.")
    ] = 256,
    hf_split: Annotated[
        str, typer.Option("--hf-split", help="HF dataset split to use when data/<task>/ is a HF dataset.")
    ] = "test",
    compile: Annotated[
        bool, typer.Option("--compile/--no-compile", help="Wrap models with torch.compile (inductor backend).")
    ] = False,
    compile_backend: Annotated[
        str, typer.Option("--compile-backend", help="torch.compile backend (inductor, aot_eager, …).")
    ] = "inductor",
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="YAML config file (overrides CLI flags)."),
    ] = None,
) -> None:
    """Run the benchmark for the selected models and tasks."""
    if config and config.exists():
        cfg = yaml.safe_load(config.read_text())
        models = cfg.get("models", models)
        tasks = cfg.get("tasks", tasks)
        data_dir = Path(cfg.get("data_dir", data_dir))
        results_dir = Path(cfg.get("results_dir", results_dir))
        max_new_tokens = cfg.get("max_new_tokens", max_new_tokens)
        compile = cfg.get("compile", compile)
        compile_backend = cfg.get("compile_backend", compile_backend)
        hf_split = cfg.get("hf_split", hf_split)

    from benchmark.runner import BenchmarkRunner

    samples_by_task = {
        task_name: _load_samples(data_dir, task_name, hf_split=hf_split)
        for task_name in tasks
    }

    runner = BenchmarkRunner(
        model_names=models,
        task_names=tasks,
        samples_by_task=samples_by_task,
        max_new_tokens=max_new_tokens,
        results_dir=results_dir,
        compile=compile,
        compile_backend=compile_backend,
    )
    runner.run()


@app.command(name="list")
def list_available() -> None:
    """List all registered models and tasks."""
    from rich.console import Console
    from rich.table import Table

    con = Console()

    model_table = Table(title="Registered Models")
    model_table.add_column("Name", style="cyan")
    model_table.add_column("Default Model ID", style="white")
    for name, mid in MODEL_REGISTRY.items():
        model_table.add_row(name, mid)
    con.print(model_table)

    task_table = Table(title="Registered Tasks")
    task_table.add_column("Name", style="cyan")
    task_table.add_column("Class", style="white")
    for name, cls in TASK_REGISTRY.items():
        task_table.add_row(name, cls.__name__)
    con.print(task_table)


if __name__ == "__main__":
    app()

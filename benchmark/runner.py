"""Benchmark runner — orchestrates models × tasks × samples."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
import pandas as pd
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from benchmark.models.base import BaseVLM
from benchmark.models.registry import load_model
from benchmark.tasks.base import BaseTask, TaskSample, TaskResult
from benchmark.tasks.registry import load_task

console = Console()


class BenchmarkRunner:
    def __init__(
        self,
        model_names: list[str],
        task_names: list[str],
        samples_by_task: dict[str, list[TaskSample]],
        max_new_tokens: int = 256,
        results_dir: str | Path = "results",
        compile: bool = False,
        compile_backend: str = "inductor",
    ) -> None:
        self.model_names = model_names
        self.task_names = task_names
        self.samples_by_task = samples_by_task
        self.max_new_tokens = max_new_tokens
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.compile = compile
        self.compile_backend = compile_backend

    # ------------------------------------------------------------------

    def run(self) -> list[TaskResult]:
        all_results: list[TaskResult] = []

        for model_name in self.model_names:
            console.rule(f"[bold cyan]Model: {model_name}")
            model = load_model(
                model_name,
                compile=self.compile,
                compile_backend=self.compile_backend,
            )

            for task_name in self.task_names:
                task = load_task(task_name)
                task_results = self._run_task(model, model_name, task, task_name)
                all_results.extend(task_results)

            model.unload()

        self._save(all_results)
        self._report(all_results)
        return all_results

    def _run_task(
        self, model: BaseVLM, model_name: str, task: BaseTask, task_name: str
    ) -> list[TaskResult]:
        samples = self.samples_by_task.get(task_name, [])
        results: list[TaskResult] = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            pid = progress.add_task(
                f"  Task [bold]{task.name}[/] — {len(samples)} samples",
                total=len(samples),
            )
            for idx, sample in enumerate(samples):
                try:
                    result = task.run(
                        model,
                        sample,
                        sample_id=str(idx),
                        max_new_tokens=self.max_new_tokens,
                    )
                except Exception as exc:
                    console.print(
                        f"  [red]ERROR[/] sample {idx} ({task_name}): {exc}"
                    )
                    result = TaskResult(
                        sample_id=str(idx),
                        model_name=model_name,
                        task_name=task_name,
                        prediction="",
                        reference=sample.reference,
                        error=str(exc),
                    )
                result.model_name = model_name
                results.append(result)
                progress.advance(pid)
        return results

    # ------------------------------------------------------------------

    def _save(self, results: list[TaskResult]) -> None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        records = [asdict(r) for r in results]

        json_path = self.results_dir / f"results_{ts}.json"
        json_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))

        csv_path = self.results_dir / f"results_{ts}.csv"
        # Flatten metrics dict into columns
        flat = []
        for r in records:
            row = {k: v for k, v in r.items() if k != "metrics"}
            row.update({f"metric_{k}": v for k, v in (r.get("metrics") or {}).items()})
            flat.append(row)
        pd.DataFrame(flat).to_csv(csv_path, index=False)

        console.print(f"\n[green]Results saved to[/] {json_path} and {csv_path}")

    def _report(self, results: list[TaskResult]) -> None:
        errored = [r for r in results if r.error]
        if errored:
            console.print(
                f"[yellow]Warning:[/] {len(errored)} sample(s) failed and are excluded from the summary."
            )
        df_rows = []
        for r in results:
            if r.error:
                continue
            row: dict = {
                "model": r.model_name,
                "task": r.task_name,
                "latency_s": r.latency_s,
            }
            row.update(r.metrics)
            df_rows.append(row)
        df = pd.DataFrame(df_rows)

        agg_cols = [c for c in df.columns if c not in ("model", "task")]
        summary = df.groupby(["model", "task"])[agg_cols].mean().reset_index()

        table = Table(title="Benchmark Summary (mean per model × task)")
        for col in summary.columns:
            table.add_column(col, style="cyan" if col in ("model", "task") else "white")
        for _, row in summary.iterrows():
            table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])
        console.print(table)

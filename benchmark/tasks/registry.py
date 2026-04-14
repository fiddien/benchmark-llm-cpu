"""Task registry."""
from __future__ import annotations

from benchmark.tasks.captioning import CaptioningTask
from benchmark.tasks.vqa import VQATask
from benchmark.tasks.structured_output import StructuredOutputTask
from benchmark.tasks.base import BaseTask

TASK_REGISTRY: dict[str, type[BaseTask]] = {
    "captioning": CaptioningTask,
    "vqa": VQATask,
    "structured_output": StructuredOutputTask,
}


def load_task(name: str) -> BaseTask:
    if name not in TASK_REGISTRY:
        raise ValueError(f"Unknown task {name!r}. Available: {list(TASK_REGISTRY)}")
    return TASK_REGISTRY[name]()

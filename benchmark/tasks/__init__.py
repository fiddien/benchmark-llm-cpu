from benchmark.tasks.base import BaseTask, TaskResult, TaskSample
from benchmark.tasks.captioning import CaptioningTask
from benchmark.tasks.vqa import VQATask
from benchmark.tasks.structured_output import StructuredOutputTask
from benchmark.tasks.registry import TASK_REGISTRY, load_task

__all__ = [
    "BaseTask",
    "TaskResult",
    "TaskSample",
    "CaptioningTask",
    "VQATask",
    "StructuredOutputTask",
    "TASK_REGISTRY",
    "load_task",
]

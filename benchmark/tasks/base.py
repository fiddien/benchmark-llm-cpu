"""Abstract base for all benchmark tasks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image

from benchmark.models.base import BaseVLM, GenerationResult


@dataclass
class TaskSample:
    image: Image.Image
    prompt: str
    reference: str | None = None  # ground-truth text (optional)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskResult:
    sample_id: str
    model_name: str
    task_name: str
    prediction: str
    reference: str | None
    metrics: dict[str, float] = field(default_factory=dict)
    latency_s: float = 0.0
    peak_memory_mb: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None  # set when generation raised an exception


class BaseTask(ABC):
    name: str  # override in subclass

    @abstractmethod
    def build_prompt(self, sample: TaskSample) -> str:
        """Return the text prompt to send to the model."""

    @abstractmethod
    def score(self, prediction: str, reference: str | None) -> dict[str, float]:
        """Compute task-specific metrics."""

    def run(
        self,
        model: BaseVLM,
        sample: TaskSample,
        sample_id: str,
        max_new_tokens: int = 256,
    ) -> TaskResult:
        prompt = self.build_prompt(sample)
        gen: GenerationResult = model.generate(
            sample.image, prompt, max_new_tokens=max_new_tokens
        )
        metrics = self.score(gen.text, sample.reference)
        return TaskResult(
            sample_id=sample_id,
            model_name=repr(model),
            task_name=self.name,
            prediction=gen.text,
            reference=sample.reference,
            metrics=metrics,
            latency_s=gen.latency_s,
            peak_memory_mb=gen.peak_memory_mb,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
        )

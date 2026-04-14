"""Image captioning task."""
from __future__ import annotations

from benchmark.tasks.base import BaseTask, TaskSample
from benchmark.metrics import bleu, rouge_l


class CaptioningTask(BaseTask):
    name = "captioning"

    DEFAULT_PROMPT = "Describe this image in detail."

    def build_prompt(self, sample: TaskSample) -> str:
        return sample.prompt or self.DEFAULT_PROMPT

    def score(self, prediction: str, reference: str | None) -> dict[str, float]:
        if reference is None:
            return {}
        return {
            "bleu4": bleu(prediction, reference),
            "rouge_l": rouge_l(prediction, reference),
        }

"""Visual Question Answering task."""
from __future__ import annotations

from benchmark.tasks.base import BaseTask, TaskSample
from benchmark.metrics import exact_match, f1_token


class VQATask(BaseTask):
    name = "vqa"

    def build_prompt(self, sample: TaskSample) -> str:
        return sample.prompt  # the question is stored in prompt

    def score(self, prediction: str, reference: str | None) -> dict[str, float]:
        if reference is None:
            return {}
        return {
            "exact_match": exact_match(prediction, reference),
            "token_f1": f1_token(prediction, reference),
        }

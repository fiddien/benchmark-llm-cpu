"""PaliGemma 2 adapter (via HuggingFace transformers)."""
from __future__ import annotations

import time
from typing import Any

from PIL import Image

from benchmark.models.base import BaseVLM, GenerationResult


class PaliGemma2Adapter(BaseVLM):
    """
    Adapter for PaliGemma 2.
    Install:  pip install transformers
    Access:   requires accepting the model licence on HuggingFace and
              running `huggingface-cli login` before first use.
    """

    model_id: str = "google/paligemma2-3b-pt-224"

    def load(self) -> None:
        import torch
        from transformers import PaliGemmaForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
        )
        self.model.eval()
        self._maybe_compile()
        self._loaded = True

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> GenerationResult:
        import torch

        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )

        t0 = time.perf_counter()
        with self._monitor_peak_rss() as mem:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    **kwargs,
                )
        latency = time.perf_counter() - t0

        # PaliGemma output includes the prompt; strip it
        trimmed = output_ids[:, inputs["input_ids"].shape[-1]:]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True
        )
        return GenerationResult(
            text=decoded[0].strip(),
            prompt_tokens=int(inputs["input_ids"].shape[-1]),
            completion_tokens=int(trimmed.shape[-1]),
            latency_s=latency,
            peak_memory_mb=mem.mb,
        )

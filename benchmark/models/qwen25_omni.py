"""Qwen2.5-Omni adapter (audio-visual model; image-only path used here)."""
from __future__ import annotations

import time
from typing import Any

from PIL import Image

from benchmark.models.base import BaseVLM, GenerationResult


class Qwen25OmniAdapter(BaseVLM):
    """
    Adapter for Qwen2.5-Omni.
    Install:  pip install transformers qwen-omni-utils[decord]
    Note: audio output is disabled; only text output is returned.
    """

    model_id: str = "Qwen/Qwen2.5-Omni-3B"

    def load(self) -> None:
        import torch
        from transformers import Qwen2_5OmniForConditionalGeneration, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,
            device_map="cpu",
            trust_remote_code=True,
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
        from qwen_omni_utils import process_mm_info

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        audios, images, video = process_mm_info(messages, use_audio_in_video=False)
        inputs = self.processor(
            text=text,
            images=images,
            audios=audios,
            return_tensors="pt",
            padding=True,
        )

        t0 = time.perf_counter()
        with self._monitor_peak_rss() as mem:
            with torch.no_grad():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    return_audio=False,
                    **kwargs,
                )
        latency = time.perf_counter() - t0

        trimmed = output[:, inputs.input_ids.shape[1]:]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return GenerationResult(
            text=decoded[0].strip(),
            prompt_tokens=int(inputs.input_ids.shape[-1]),
            completion_tokens=int(trimmed.shape[-1]),
            latency_s=latency,
            peak_memory_mb=mem.mb,
        )

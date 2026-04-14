"""Qwen3-VL adapter (via HuggingFace transformers)."""
from __future__ import annotations

import time
from typing import Any

from PIL import Image

from benchmark.models.base import BaseVLM, GenerationResult


class Qwen3VLAdapter(BaseVLM):
    """
    Adapter for Qwen3-VL / Qwen2.5-VL family.
    Uses the qwen-vl-utils helper for image/video message construction.
    Install:  pip install transformers qwen-vl-utils
    """

    model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    def load(self) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5VLForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        self.model = Qwen2_5VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32,  # CPU: no bfloat16 support on all CPUs
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
        from qwen_vl_utils import process_vision_info

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
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
        )

        t0 = time.perf_counter()
        with self._monitor_peak_rss() as mem:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, **kwargs
                )
        latency = time.perf_counter() - t0

        # Trim the input tokens from the output
        trimmed = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        decoded = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return GenerationResult(
            text=decoded[0].strip(),
            prompt_tokens=int(inputs.input_ids.shape[-1]),
            completion_tokens=int(output_ids.shape[-1] - inputs.input_ids.shape[-1]),
            latency_s=latency,
            peak_memory_mb=mem.mb,
        )

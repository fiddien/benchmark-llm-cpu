"""Moondream 3 adapter (uses the `moondream` PyPI package)."""
from __future__ import annotations

import time
from typing import Any

from PIL import Image

from benchmark.models.base import BaseVLM, GenerationResult


class MoondreamAdapter(BaseVLM):
    model_id: str = "vikhyatk/moondream2"

    def load(self) -> None:
        import moondream as md  # pip install moondream

        self.model = md.vl(model=self.model_id)
        self._maybe_compile()  # no-op: moondream model is not an nn.Module
        self._loaded = True

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> GenerationResult:
        t0 = time.perf_counter()
        with self._monitor_peak_rss() as mem:
            encoded = self.model.encode_image(image)
            result = self.model.query(encoded, prompt)
        latency = time.perf_counter() - t0

        # moondream returns {"answer": "..."} or a plain string depending on version
        if isinstance(result, dict):
            text = result.get("answer", str(result))
        else:
            text = str(result)

        return GenerationResult(
            text=text,
            latency_s=latency,
            peak_memory_mb=mem.mb,
        )

"""Abstract base class for all VLM adapters."""
from __future__ import annotations

import time
import threading
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Generator

from PIL import Image


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    peak_memory_mb: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class _PeakMemHolder:
    """Mutable holder populated by _monitor_peak_rss after the block exits."""
    mb: float = 0.0


class BaseVLM(ABC):
    """Common interface every VLM adapter must implement."""

    model_id: str  # HuggingFace repo id or local path

    def __init__(
        self,
        model_id: str | None = None,
        compile: bool = False,
        compile_backend: str = "inductor",
        **kwargs: Any,
    ) -> None:
        if model_id is not None:
            self.model_id = model_id
        self._compile = compile
        self._compile_backend = compile_backend
        self._kwargs = kwargs
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self) -> None:
        """Download weights and initialise the model on CPU."""

    def unload(self) -> None:
        """Release model weights from memory (optional override)."""
        import gc
        import torch

        for attr in ("model", "processor", "tokenizer"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()
        self._loaded = False

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    @abstractmethod
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs: Any,
    ) -> GenerationResult:
        """Run a single image+text forward pass and return a GenerationResult."""

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def caption(self, image: Image.Image, **kwargs: Any) -> GenerationResult:
        return self.generate(image, "Describe this image in detail.", **kwargs)

    def answer(
        self, image: Image.Image, question: str, **kwargs: Any
    ) -> GenerationResult:
        return self.generate(image, question, **kwargs)

    def structured(
        self, image: Image.Image, schema_prompt: str, **kwargs: Any
    ) -> GenerationResult:
        """Ask the model to respond according to a JSON schema."""
        return self.generate(image, schema_prompt, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _timed(fn):
        """Decorator: wraps a callable and returns (result, elapsed_s)."""
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            return out, time.perf_counter() - t0
        return wrapper

    def _maybe_compile(self) -> None:
        """Wrap self.model with torch.compile if requested.

        Only applied when self.model is a proper nn.Module — adapters that use
        third-party objects (e.g. moondream) are skipped automatically.
        """
        if not self._compile:
            return
        import torch
        model = getattr(self, "model", None)
        if model is None or not isinstance(model, torch.nn.Module):
            return
        self.model = torch.compile(self.model, backend=self._compile_backend)

    @contextmanager
    def _monitor_peak_rss(
        self, interval_s: float = 0.05
    ) -> Generator["_PeakMemHolder", None, None]:
        """Context manager that continuously samples RSS during the block.

        Usage::

            with self._monitor_peak_rss() as mem:
                # ... run inference ...
            peak_mb = mem.mb
        """
        import psutil, os

        proc = psutil.Process(os.getpid())
        baseline = proc.memory_info().rss
        peak_rss = [baseline]
        stop = threading.Event()

        def _sample() -> None:
            while not stop.wait(interval_s):
                try:
                    peak_rss[0] = max(peak_rss[0], proc.memory_info().rss)
                except Exception:
                    pass

        t = threading.Thread(target=_sample, daemon=True)
        t.start()
        holder = _PeakMemHolder()
        try:
            yield holder
        finally:
            stop.set()
            t.join(timeout=1.0)
            holder.mb = max(peak_rss[0] - baseline, 0) / 1024 / 1024

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_id={self.model_id!r})"

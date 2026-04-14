"""Central model registry."""
from __future__ import annotations

from typing import Any

from benchmark.models.base import BaseVLM

# Registry maps short name -> (adapter_class_path, default_model_id)
_REGISTRY: dict[str, tuple[str, str]] = {
    # "moondream3": (
    #     "benchmark.models.moondream.MoondreamAdapter",
    #     "vikhyatk/moondream2",  # update when moondream3 is publicly released
    # ),
    "qwen3-vl": (
        "benchmark.models.qwen3_vl.Qwen3VLAdapter",
        "Qwen/Qwen2.5-VL-3B-Instruct",  # placeholder; swap for Qwen3-VL when available
    ),
    "qwen2.5-omni": (
        "benchmark.models.qwen25_omni.Qwen25OmniAdapter",
        "Qwen/Qwen2.5-Omni-3B",
    ),
    "paligemma2": (
        "benchmark.models.paligemma2.PaliGemma2Adapter",
        "google/paligemma2-3b-pt-224",
    ),
}

# Public view: name -> default_model_id
MODEL_REGISTRY: dict[str, str] = {k: v[1] for k, v in _REGISTRY.items()}


def _import_class(dotted: str):
    module_path, cls_name = dotted.rsplit(".", 1)
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, cls_name)


def load_model(name: str, model_id: str | None = None, **kwargs: Any) -> BaseVLM:
    """Instantiate and load a model by its short registry name."""
    if name not in _REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(_REGISTRY)}"
        )
    cls_path, default_id = _REGISTRY[name]
    cls = _import_class(cls_path)
    instance: BaseVLM = cls(model_id=model_id or default_id, **kwargs)
    instance.load()
    return instance

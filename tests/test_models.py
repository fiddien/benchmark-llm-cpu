"""Smoke tests for model registry (no weight downloads)."""
from __future__ import annotations

import pytest

from benchmark.models.registry import MODEL_REGISTRY, _REGISTRY


def test_all_registry_names_present():
    expected = {"moondream3", "qwen3-vl", "qwen2.5-omni", "paligemma2"}
    assert expected == set(MODEL_REGISTRY.keys())


def test_registry_class_paths_importable():
    """Verify that adapter modules can be imported (not that weights load)."""
    import importlib
    for name, (cls_path, _) in _REGISTRY.items():
        module_path, cls_name = cls_path.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        assert hasattr(mod, cls_name), f"{cls_name} not found in {module_path}"


def test_load_model_unknown_raises():
    from benchmark.models.registry import load_model
    with pytest.raises(ValueError, match="Unknown model"):
        load_model("does-not-exist")

"""Unit tests — no model weights required."""
from __future__ import annotations

import json

import pytest
from PIL import Image

from benchmark.tasks.base import TaskSample
from benchmark.tasks.captioning import CaptioningTask
from benchmark.tasks.vqa import VQATask
from benchmark.tasks.structured_output import StructuredOutputTask
from benchmark.metrics import bleu, rouge_l, exact_match, f1_token


# ---------------------------------------------------------------------------
# Metric tests
# ---------------------------------------------------------------------------

def test_bleu_identical():
    assert bleu("the cat sat", "the cat sat") == pytest.approx(1.0, abs=1e-3)


def test_bleu_no_overlap():
    assert bleu("foo bar baz", "qux quux corge") == 0.0


def test_rouge_l_identical():
    assert rouge_l("hello world", "hello world") == pytest.approx(1.0, abs=1e-3)


def test_exact_match():
    assert exact_match("Yes", "yes") == 1.0
    assert exact_match("No", "yes") == 0.0


def test_f1_partial():
    score = f1_token("the quick brown fox", "the quick fox")
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# Task build_prompt / score tests (no model needed)
# ---------------------------------------------------------------------------

def _dummy_sample(prompt="", reference=None, metadata=None):
    img = Image.new("RGB", (64, 64))
    return TaskSample(image=img, prompt=prompt, reference=reference, metadata=metadata or {})


def test_captioning_default_prompt():
    task = CaptioningTask()
    s = _dummy_sample(prompt="")
    assert "Describe" in task.build_prompt(s)


def test_captioning_score_no_reference():
    task = CaptioningTask()
    assert task.score("anything", None) == {}


def test_vqa_score():
    task = VQATask()
    metrics = task.score("42", "42")
    assert metrics["exact_match"] == 1.0
    assert metrics["token_f1"] == pytest.approx(1.0)


def test_structured_output_valid_json():
    task = StructuredOutputTask()
    good = json.dumps({"name": "Alice", "age": "30"})
    assert task.score(good, None)["valid_json"] == 1.0


def test_structured_output_invalid_json():
    task = StructuredOutputTask()
    assert task.score("not json at all", None)["valid_json"] == 0.0


def test_structured_output_field_match():
    task = StructuredOutputTask()
    ref = json.dumps({"total": "12.99", "vendor": "Acme"})
    pred = json.dumps({"total": "12.99", "vendor": "Acme"})
    metrics = task.score(pred, ref)
    assert metrics["field_match"] == pytest.approx(1.0)


def test_structured_prompt_contains_schema():
    task = StructuredOutputTask()
    schema = '{"type": "object"}'
    s = _dummy_sample(prompt="Extract data.", metadata={"schema": schema})
    prompt = task.build_prompt(s)
    assert schema in prompt
    assert "Extract data." in prompt

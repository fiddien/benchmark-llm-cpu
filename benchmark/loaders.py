"""Alternative data loaders for benchmark tasks.

Supports loading samples from HuggingFace datasets saved with
``dataset.save_to_disk()``, in addition to the default JSON-manifest format.
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image

from benchmark.tasks.base import TaskSample


def load_hf_dataset(
    dataset_path: str | Path,
    split: str = "test",
) -> list[TaskSample]:
    """Load TaskSamples from a HuggingFace dataset saved with save_to_disk().

    Expected dataset schema (from vlm-fine-tuning-dataset):
    - ``images``      : list of PIL images (one per row)
    - ``messages``    : full conversation — system / user / assistant turns;
                        the assistant turn is used as the ground-truth reference.
    - ``only_images`` : user turn only (prompt text + image placeholder);
                        the user text is used as the benchmark prompt.

    The prompt extracted from ``only_images`` already contains complete
    instructions and an embedded schema, so the schema-hint that
    ``StructuredOutputTask.build_prompt`` would normally append is suppressed
    via ``metadata={"schema_hint": ""}``.

    Args:
        dataset_path: Path to the dataset directory (contains dataset_dict.json).
        split: Which split to load — ``"train"``, ``"val"``, or ``"test"``.

    Returns:
        List of ``TaskSample`` objects ready for ``StructuredOutputTask``.
    """
    from datasets import load_from_disk  # lazy import — not always installed

    ds = load_from_disk(str(dataset_path))
    if split not in ds:
        available = list(ds.keys())
        raise ValueError(f"Split {split!r} not found. Available: {available}")

    split_ds = ds[split]
    table = split_ds.data.table  # raw pyarrow table; avoids image-decode errors

    samples: list[TaskSample] = []
    for i in range(table.num_rows):
        row: dict[str, Any] = table.slice(i, 1).to_pydict()

        image = _decode_first_image(row["images"][0])
        prompt = _extract_user_text(row["only_images"][0])
        reference = _extract_assistant_text(row["messages"][0])

        samples.append(
            TaskSample(
                image=image,
                prompt=prompt,
                reference=reference,
                # The prompt already contains full instructions + schema.
                # Setting schema_hint="" prevents StructuredOutputTask from
                # appending a redundant second hint.
                metadata={"schema_hint": ""},
            )
        )

    return samples


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _decode_first_image(images_cell: list[dict]) -> Image.Image:
    """Decode the first image in a raw Arrow images cell to a PIL Image."""
    img_dict = images_cell[0]
    raw_bytes: bytes | None = img_dict.get("bytes")
    path: str | None = img_dict.get("path")

    if raw_bytes:
        return Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    if path:
        return Image.open(path).convert("RGB")
    raise ValueError(f"Cannot decode image — no bytes or path found: {img_dict!r}")


def _extract_user_text(messages: list[dict]) -> str:
    """Return the text content of the first user message."""
    for msg in messages:
        if msg["role"] == "user":
            for content in msg["content"]:
                if content["type"] == "text" and content["text"]:
                    return content["text"]
    return ""


def _extract_assistant_text(messages: list[dict]) -> str | None:
    """Return the text content of the first assistant message, or None."""
    for msg in messages:
        if msg["role"] == "assistant":
            for content in msg["content"]:
                if content["type"] == "text" and content["text"]:
                    return content["text"]
    return None

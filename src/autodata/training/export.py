"""Make a trained model directory serve on its own.

On the run box the GRPO save writes the weights and part of the processor
but no tokenizer files, and vLLM cannot load the directory. Training does
not change the tokenizer, so the files the base model has and the export
lacks are copied over. Weights are never copied.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)

_WEIGHT_SUFFIXES = (".safetensors", ".safetensors.index.json", ".bin", ".pt", ".pth")


def complete_export(base_dir: str | Path, out_dir: str | Path) -> list[str]:
    """Copy non-weight files from ``base_dir`` that ``out_dir`` lacks; return their names."""
    base, out = Path(base_dir), Path(out_dir)
    copied = []
    for src in sorted(base.iterdir()):
        if not src.is_file() or src.name.endswith(_WEIGHT_SUFFIXES) or (out / src.name).exists():
            continue
        shutil.copy2(src, out / src.name)
        copied.append(src.name)
    if copied:
        logger.info("copied from %s into %s: %s", base, out, ", ".join(copied))
    return copied


__all__ = ["complete_export"]

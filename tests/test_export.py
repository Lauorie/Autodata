"""A trained model directory must serve on its own.

The GRPO save writes weights and part of the processor but, on the run box,
no tokenizer files, so vLLM could not load ``final/``. The tokenizer does
not change in training: the files the base directory has and the export
lacks are copied over, weights never.
"""

from __future__ import annotations

from pathlib import Path

from autodata.training import complete_export


def _files(d: Path, names: dict[str, str]) -> Path:
    d.mkdir()
    for name, text in names.items():
        (d / name).write_text(text)
    return d


def test_missing_tokenizer_files_come_from_the_base(tmp_path: Path) -> None:
    base = _files(tmp_path / "base", {"tokenizer.json": "tok", "vocab.json": "v",
                                      "config.json": "base-cfg"})
    out = _files(tmp_path / "final", {"config.json": "trained-cfg"})
    assert complete_export(base, out) == ["tokenizer.json", "vocab.json"]
    assert (out / "tokenizer.json").read_text() == "tok"
    assert (out / "config.json").read_text() == "trained-cfg"


def test_base_weights_are_never_copied(tmp_path: Path) -> None:
    base = _files(tmp_path / "base", {"model.safetensors-00001-of-00002.safetensors": "w",
                                      "model.safetensors.index.json": "{}",
                                      "merges.txt": "m"})
    out = _files(tmp_path / "final", {"model.safetensors": "trained"})
    assert complete_export(base, out) == ["merges.txt"]

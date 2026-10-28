"""Build the GRPO training set and the held-out test set from finished runs.

Reads only <run>/final_qa.jsonl: questions the loop accepted and the
end-of-loop filter kept (Sec 3.1). Rejected rounds are never training data.
Repeated questions are dropped, then whole papers go to the test set until
it holds data.test_size questions (100 in the paper).

Writes, under <outputs_root>/<run_id>/data:
    train.jsonl, test.jsonl
    manifest.json   source runs, counts, seed, test papers, file hashes

Usage:
    python scripts/build_dataset.py run_id=data1 \\
        'data.runs=[/root/autodl-fs/autodata/outputs/full_a,/root/autodl-fs/autodata/outputs/full_b]'
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import hydra
from omegaconf import DictConfig

from autodata.data_module.qa_dataset import dedupe, load_final_qas, split_by_paper
from autodata.utils import dump_jsonl, get_logger, setup_logging, write_json

logger = get_logger(__name__)


def _sha12(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    runs = [str(r) for r in cfg.data.runs]
    if not runs:
        raise SystemExit("set data.runs to the run directories to read")
    loaded = load_final_qas(runs)
    qas = dedupe(loaded)
    train, test = split_by_paper(qas, int(cfg.data.test_size), int(cfg.seed))

    out = Path(cfg.paths.outputs_root) / cfg.run_id / "data"
    files = {"train": out / "train.jsonl", "test": out / "test.jsonl"}
    dump_jsonl(files["train"], (q.model_dump() for q in train))
    dump_jsonl(files["test"], (q.model_dump() for q in test))
    write_json(out / "manifest.json", {
        "runs": runs,
        "seed": int(cfg.seed),
        "loaded": len(loaded),
        "after_dedupe": len(qas),
        "train": len(train),
        "test": len(test),
        "test_papers": sorted({q.paper_id for q in test}),
        "sha256_12": {k: _sha12(p) for k, p in files.items()},
    })
    logger.info("loaded %d, %d after dedupe: %d train, %d test -> %s",
                len(loaded), len(qas), len(train), len(test), out)


if __name__ == "__main__":
    main()

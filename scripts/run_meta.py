"""Meta-optimize the challenger and verifier prompts (arXiv 2606.25996, Sec 4).

Writes, under <outputs_root>/<run_id>/meta:
    split.json          train and validation paper ids
    history.jsonl       one record per iteration
    population.json     kept variants with their validation scores
    variants/<id>/      prompts/, diff.patch against the seed, variant.json
    iterations/<n>/     train and validation trajectories, analysis.md
    usage.jsonl         one line per LLM call
    summary.json        best variant and token totals

Usage (from repo root, on remote):
    python scripts/run_meta.py run_id=meta1 models=flash meta.iterations=2
"""

from __future__ import annotations

import random
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from autodata.config_runtime import (
    acceptance_from_dictconfig,
    build_client_pool,
    inner_loop_from_dictconfig,
    meta_from_dictconfig,
    role_configs_from_dictconfig,
)
from autodata.data_module import iter_papers
from autodata.llm import PromptSet, UsageMeter
from autodata.meta import run_meta_optimization, split_papers
from autodata.utils import get_logger, set_seed, setup_logging, write_json

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    load_dotenv(dotenv_path=".secrets/remote.env", override=False)
    set_seed(cfg.seed)

    out_dir = Path(cfg.paths.outputs_root) / cfg.run_id / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    meter = UsageMeter(out_dir / "usage.jsonl")
    clients = build_client_pool(role_configs_from_dictconfig(cfg.models, cfg.solver_generation),
                                meter=meter)

    papers = list(iter_papers(cfg.paths.papers_dir, limit=cfg.run.papers_limit))
    train, val = split_papers(papers, int(cfg.meta.train_papers), int(cfg.meta.val_papers),
                              int(cfg.seed))
    write_json(out_dir / "split.json", {"train": [p.paper_id for p in train],
                                        "val": [p.paper_id for p in val]})
    logger.info("papers: %d train, %d validation (of %d)", len(train), len(val), len(papers))

    seed = PromptSet.load(cfg.prompts_dir) if cfg.prompts_dir else PromptSet.seed()
    population = run_meta_optimization(
        seed, train, val, clients, acceptance_from_dictconfig(cfg.acceptance),
        inner_loop_from_dictconfig(cfg.inner_loop), meta_from_dictconfig(cfg.meta),
        out_dir, random.Random(cfg.seed))

    best = max(population, key=lambda v: v.mean if v.mean is not None else -1.0)
    write_json(out_dir / "summary.json", {
        "run_id": cfg.run_id,
        "best": {"variant_id": best.variant_id, "scores": list(best.scores), "mean": best.mean},
        "population": [v.variant_id for v in population],
        "usage": meter.totals(),
    })
    logger.info("DONE meta: population=%d best=%s mean=%s",
                len(population), best.variant_id, best.mean)


if __name__ == "__main__":
    main()

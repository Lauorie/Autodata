"""Run the meta-optimization outer loop on a held-out validation set.

Usage:
    python scripts/run_meta.py run_id=meta1 meta.n_generations=4 meta.validation_chunks=8
"""

from __future__ import annotations

import logging
import random
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from autodata.config_runtime import (
    build_client_pool,
    role_configs_from_dictconfig,
)
from autodata.data_module import chunk_paper, iter_papers
from autodata.meta import MetaConfig, run_meta_optimization
from autodata.pipeline.acceptance import AcceptanceCriteria
from autodata.pipeline.inner_loop import InnerLoopConfig
from autodata.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


def _criteria_from(cfg: DictConfig) -> AcceptanceCriteria:
    return AcceptanceCriteria(
        quality_verifier_must_pass=bool(cfg.quality_verifier_must_pass),
        weak_avg_max=float(cfg.weak_avg_max),
        weak_max_max=float(cfg.weak_max_max),
        no_zeros=bool(cfg.no_zeros),
        strong_avg_min=float(cfg.strong_avg_min),
        strong_avg_max=float(cfg.strong_avg_max),
        gap_min=float(cfg.gap_min),
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    load_dotenv(dotenv_path=".secrets/remote.env", override=False)
    set_seed(cfg.seed)

    out_dir = Path(cfg.paths.outputs_root) / cfg.run_id / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)

    role_configs = role_configs_from_dictconfig(cfg.models)
    clients = build_client_pool(role_configs)
    criteria = _criteria_from(cfg.acceptance)
    inner_cfg = InnerLoopConfig(
        max_iterations=int(cfg.inner_loop.max_iterations),
        num_solver_samples=int(cfg.inner_loop.num_solver_samples),
        qa_per_chunk=int(cfg.inner_loop.qa_per_chunk),
        parallel_solvers=bool(cfg.inner_loop.get("parallel_solvers", True)),
    )
    n_gen = int(cfg.get("meta", {}).get("n_generations", 4))
    n_val = int(cfg.get("meta", {}).get("validation_chunks", 8))
    temp = float(cfg.get("meta", {}).get("temperature", 0.1))

    # Build a tiny held-out chunk set drawn deterministically from the corpus
    rng = random.Random(cfg.seed)
    all_chunks = []
    for p in iter_papers(cfg.paths.papers_dir, limit=cfg.run.papers_limit):
        all_chunks.extend(chunk_paper(p, max_chars=int(cfg.inner_loop.max_chunk_chars)))
    rng.shuffle(all_chunks)
    val_chunks = all_chunks[:n_val]
    logger.info("validation set: %d chunks (out of %d)", len(val_chunks), len(all_chunks))

    seed_prompts = Path(__file__).resolve().parent.parent / "src" / "autodata" / "llm" / "prompts.py"
    pop = run_meta_optimization(
        seed_prompts_path=seed_prompts,
        val_chunks=val_chunks,
        clients=clients,
        criteria=criteria,
        inner_cfg=inner_cfg,
        out_dir=out_dir,
        cfg=MetaConfig(n_generations=n_gen, temperature=temp, validation_chunks=n_val),
        rng=rng,
    )
    logger.info("DONE meta: pop_size=%d best_score=%.3f",
                len(pop), max(v.score for v in pop))


if __name__ == "__main__":
    main()

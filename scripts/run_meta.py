"""Run the meta-optimization outer loop on a held-out validation set.

Usage:
    python scripts/run_meta.py run_id=meta1 meta.n_generations=4 meta.validation_papers=8
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
from autodata.data_module import iter_papers
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

    role_configs = role_configs_from_dictconfig(cfg.models, cfg.solver_generation)
    clients = build_client_pool(role_configs)
    criteria = _criteria_from(cfg.acceptance)
    inner_cfg = InnerLoopConfig(
        max_iterations=int(cfg.inner_loop.max_iterations),
        num_solver_samples=int(cfg.inner_loop.num_solver_samples),
        infra_retries=int(cfg.inner_loop.infra_retries),
        parallel_solvers=bool(cfg.inner_loop.get("parallel_solvers", True)),
    )
    n_gen = int(cfg.get("meta", {}).get("n_generations", 4))
    n_val = int(cfg.get("meta", {}).get("validation_papers", 8))
    temp = float(cfg.get("meta", {}).get("temperature", 0.1))

    # Build a tiny held-out paper set drawn deterministically from the corpus
    rng = random.Random(cfg.seed)
    all_papers = list(iter_papers(cfg.paths.papers_dir, limit=cfg.run.papers_limit))
    rng.shuffle(all_papers)
    val_papers = all_papers[:n_val]
    logger.info("validation set: %d papers (out of %d)", len(val_papers), len(all_papers))

    seed_prompts = Path(__file__).resolve().parent.parent / "src" / "autodata" / "llm" / "prompts.py"
    pop = run_meta_optimization(
        seed_prompts_path=seed_prompts,
        val_papers=val_papers,
        clients=clients,
        criteria=criteria,
        inner_cfg=inner_cfg,
        out_dir=out_dir,
        cfg=MetaConfig(n_generations=n_gen, temperature=temp, validation_papers=n_val),
        rng=rng,
    )
    logger.info("DONE meta: pop_size=%d best_score=%.3f",
                len(pop), max(v.score for v in pop))


if __name__ == "__main__":
    main()

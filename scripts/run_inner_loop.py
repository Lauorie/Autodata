"""Run the Agentic Self-Instruct inner loop over all chunks of all papers.

Usage (from repo root, on remote):
    python scripts/run_inner_loop.py run_id=smoke run.papers_limit=5
    python scripts/run_inner_loop.py run_id=full
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from autodata.config_runtime import (
    build_client_pool,
    role_configs_from_dictconfig,
)
from autodata.data_module import chunk_paper, iter_papers
from autodata.pipeline.acceptance import AcceptanceCriteria
from autodata.pipeline.inner_loop import InnerLoopConfig, run_inner_loop_for_chunk
from autodata.utils import get_logger, set_seed, setup_logging, write_json

logger = get_logger(__name__)


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

    out_dir = Path(cfg.paths.outputs_root) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run_id=%s out_dir=%s", cfg.run_id, out_dir)

    role_configs = role_configs_from_dictconfig(cfg.models)
    clients = build_client_pool(role_configs)
    criteria = _criteria_from(cfg.acceptance)
    inner_cfg = InnerLoopConfig(
        max_iterations=int(cfg.inner_loop.max_iterations),
        num_solver_samples=int(cfg.inner_loop.num_solver_samples),
        qa_per_chunk=int(cfg.inner_loop.qa_per_chunk),
        parallel_solvers=bool(cfg.inner_loop.get("parallel_solvers", True)),
    )

    accepted_qas: list[dict[str, Any]] = []
    all_rounds_path = out_dir / "rounds.jsonl"
    accepted_path = out_dir / "accepted_qa.jsonl"
    summary_path = out_dir / "summary.json"

    n_chunks = 0
    n_accepted = 0
    n_rounds = 0
    t_start = time.time()

    chunks_per_paper = cfg.inner_loop.get("chunks_per_paper", None)
    chunk_concurrency = int(cfg.run.get("chunk_concurrency", 1))
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    write_lock = Lock()

    with all_rounds_path.open("w") as f_rounds, accepted_path.open("w") as f_acc:
        papers = list(iter_papers(cfg.paths.papers_dir, limit=cfg.run.papers_limit))

        all_chunks = []
        for paper in papers:
            cks = chunk_paper(paper, max_chars=int(cfg.inner_loop.max_chunk_chars))
            if chunks_per_paper is not None:
                cks = cks[:int(chunks_per_paper)]
            all_chunks.extend(cks)
        logger.info("processing %d papers -> %d chunks (concurrency=%d)",
                    len(papers), len(all_chunks), chunk_concurrency)

        def _write_one(chunk, qa, rounds):
            nonlocal n_rounds, n_accepted
            with write_lock:
                for r in rounds:
                    n_rounds += 1
                    f_rounds.write(json.dumps({
                        "paper_id": chunk.paper_id,
                        "chunk_id": chunk.chunk_id,
                        "section": chunk.section,
                        **r.__dict__,
                    }, default=str, ensure_ascii=False) + "\n")
                    f_rounds.flush()
                if qa is not None:
                    n_accepted += 1
                    f_acc.write(json.dumps(qa.model_dump(), ensure_ascii=False) + "\n")
                    f_acc.flush()

        with ThreadPoolExecutor(max_workers=chunk_concurrency) as pool:
            futures = {pool.submit(run_inner_loop_for_chunk, ck, clients, criteria, inner_cfg): ck
                       for ck in all_chunks}
            for fut in as_completed(futures):
                chunk = futures[fut]
                n_chunks += 1
                try:
                    qa, rounds = fut.result()
                except Exception as e:
                    logger.exception("chunk %s (paper=%s) crashed: %s",
                                     chunk.chunk_id, chunk.paper_id, e)
                    # Persist a failure marker so the run summary reflects it.
                    with write_lock:
                        f_rounds.write(json.dumps({
                            "paper_id": chunk.paper_id,
                            "chunk_id": chunk.chunk_id,
                            "section": chunk.section,
                            "error": f"{type(e).__name__}: {e}",
                            "accepted": False,
                        }, ensure_ascii=False) + "\n")
                        f_rounds.flush()
                    continue
                _write_one(chunk, qa, rounds)
                if n_chunks % 5 == 0 or qa is not None:
                    logger.info("progress: %d/%d chunks, %d accepted (%.1f%%)",
                                n_chunks, len(all_chunks), n_accepted,
                                100 * n_accepted / max(n_chunks, 1))

    elapsed = time.time() - t_start
    write_json(summary_path, {
        "run_id": cfg.run_id,
        "papers": len(papers),
        "chunks": n_chunks,
        "rounds": n_rounds,
        "accepted_qas": n_accepted,
        "elapsed_s": elapsed,
        "yield_per_chunk": n_accepted / max(n_chunks, 1),
        "rounds_per_chunk": n_rounds / max(n_chunks, 1),
    })
    logger.info("DONE: chunks=%d rounds=%d accepted=%d (%.1f%%) elapsed=%.1fs",
                n_chunks, n_rounds, n_accepted,
                100 * n_accepted / max(n_chunks, 1), elapsed)


if __name__ == "__main__":
    main()

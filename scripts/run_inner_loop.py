"""Run the Agentic Self-Instruct inner loop over every paper (one task per paper).

Writes, under the run directory:
    rounds.jsonl        one row per round, with its outcome
    accepted_qa.jsonl   questions the loop accepted (the paper's 2.8k stage)
    final_filter.jsonl  end-of-loop filter verdict per accepted question
    final_qa.jsonl      questions that also pass the filter (the 1.3k stage)
    summary.json        counts: outcomes per round, solver truncation rates

Usage (from repo root, on remote):
    python scripts/run_inner_loop.py run_id=smoke run.papers_limit=5
    python scripts/run_inner_loop.py run_id=full
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from autodata.config_runtime import (
    build_client_pool,
    role_configs_from_dictconfig,
)
from autodata.data_module import Paper, iter_papers
from autodata.llm import LLMClient, PromptSet, QAPair, UsageMeter, VerifierVerdict
from autodata.pipeline.acceptance import AcceptanceCriteria
from autodata.pipeline.final_filter import RubricShape, final_filter
from autodata.pipeline.inner_loop import InnerLoopConfig, RoundRecord, run_inner_loop_for_paper
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


def _filter_accepted(paper: Paper, qa: QAPair, qv: LLMClient,
                     shape: RubricShape, retries: int) -> VerifierVerdict:
    """End-of-loop filter. A verifier outage fails closed: the question stays
    in accepted_qa.jsonl and can be filtered again later."""
    try:
        return final_filter(paper, qa, qv, shape, retries)
    except Exception as e:
        logger.exception("final filter failed for %s: %s", paper.paper_id, e)
        return VerifierVerdict(passed=False, feedback=f"{type(e).__name__}: {e}",
                               issues=["final_filter_error"])


def _process(paper: Paper, clients: dict[str, LLMClient], criteria: AcceptanceCriteria,
             inner_cfg: InnerLoopConfig, shape: RubricShape, prompts: PromptSet,
             ) -> tuple[QAPair | None, list[RoundRecord], VerifierVerdict | None]:
    qa, rounds = run_inner_loop_for_paper(paper, clients, criteria, inner_cfg, prompts)
    verdict = (_filter_accepted(paper, qa, clients["quality_verifier"], shape,
                                inner_cfg.infra_retries)
               if qa is not None else None)
    return qa, rounds, verdict


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    load_dotenv(dotenv_path=".secrets/remote.env", override=False)
    set_seed(cfg.seed)

    out_dir = Path(cfg.paths.outputs_root) / cfg.run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("run_id=%s out_dir=%s", cfg.run_id, out_dir)

    role_configs = role_configs_from_dictconfig(cfg.models, cfg.solver_generation)
    meter = UsageMeter(out_dir / "usage.jsonl")     # one line per completed call
    clients = build_client_pool(role_configs, meter=meter)
    criteria = _criteria_from(cfg.acceptance)
    inner_cfg = InnerLoopConfig(
        max_iterations=int(cfg.inner_loop.max_iterations),
        num_solver_samples=int(cfg.inner_loop.num_solver_samples),
        infra_retries=int(cfg.inner_loop.infra_retries),
        parallel_solvers=bool(cfg.inner_loop.get("parallel_solvers", True)),
    )
    shape = RubricShape(**cfg.final_filter)
    prompts = PromptSet.load(cfg.prompts_dir) if cfg.prompts_dir else PromptSet.seed()
    logger.info("prompts: %s", cfg.prompts_dir or "seed")

    all_rounds_path = out_dir / "rounds.jsonl"
    accepted_path = out_dir / "accepted_qa.jsonl"
    verdicts_path = out_dir / "final_filter.jsonl"
    final_path = out_dir / "final_qa.jsonl"
    summary_path = out_dir / "summary.json"

    n_done = 0
    n_accepted = 0
    n_final = 0
    n_rounds = 0
    outcomes: Counter[str] = Counter()
    # Solver samples graded / cut at the token budget (paper App A, Table 8).
    samples: Counter[str] = Counter()
    t_start = time.time()

    paper_concurrency = int(cfg.run.get("paper_concurrency", 1))
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    write_lock = Lock()

    with all_rounds_path.open("w") as f_rounds, accepted_path.open("w") as f_acc, \
            verdicts_path.open("w") as f_verdicts, final_path.open("w") as f_final:
        papers = list(iter_papers(cfg.paths.papers_dir, limit=cfg.run.papers_limit))
        logger.info("processing %d papers (concurrency=%d)", len(papers), paper_concurrency)

        def _write_one(paper, qa, rounds, verdict):
            nonlocal n_rounds, n_accepted, n_final
            with write_lock:
                for r in rounds:
                    n_rounds += 1
                    outcomes[r.outcome] += 1
                    for solver, rows in (("weak", r.weak_samples), ("strong", r.strong_samples)):
                        samples[solver] += len(rows)
                        samples[f"{solver}_truncated"] += sum(s["truncated"] for s in rows)
                    f_rounds.write(json.dumps({
                        "paper_id": paper.paper_id,
                        **r.__dict__,
                    }, default=str, ensure_ascii=False) + "\n")
                    f_rounds.flush()
                if qa is None:
                    return
                n_accepted += 1
                f_acc.write(json.dumps(qa.model_dump(), ensure_ascii=False) + "\n")
                f_acc.flush()
                f_verdicts.write(json.dumps({"paper_id": paper.paper_id, **verdict.model_dump()},
                                            ensure_ascii=False) + "\n")
                f_verdicts.flush()
                if verdict.passed:
                    n_final += 1
                    f_final.write(json.dumps(qa.model_dump(), ensure_ascii=False) + "\n")
                    f_final.flush()

        with ThreadPoolExecutor(max_workers=paper_concurrency) as pool:
            futures = {pool.submit(_process, p, clients, criteria, inner_cfg, shape, prompts): p
                       for p in papers}
            for fut in as_completed(futures):
                paper = futures[fut]
                n_done += 1
                try:
                    qa, rounds, verdict = fut.result()
                except Exception as e:
                    logger.exception("paper %s crashed: %s", paper.paper_id, e)
                    # Persist a failure marker so the run summary reflects it.
                    with write_lock:
                        f_rounds.write(json.dumps({
                            "paper_id": paper.paper_id,
                            "error": f"{type(e).__name__}: {e}",
                            "accepted": False,
                        }, ensure_ascii=False) + "\n")
                        f_rounds.flush()
                    continue
                _write_one(paper, qa, rounds, verdict)
                if n_done % 5 == 0 or qa is not None:
                    logger.info("progress: %d/%d papers, %d accepted, %d kept after filter",
                                n_done, len(papers), n_accepted, n_final)

    elapsed = time.time() - t_start
    write_json(summary_path, {
        "run_id": cfg.run_id,
        "prompts_dir": cfg.prompts_dir,
        "papers": len(papers),
        "rounds": n_rounds,
        "accepted_qas": n_accepted,
        "final_qas": n_final,
        "outcomes": dict(outcomes),
        "weak_truncation_rate": samples["weak_truncated"] / max(samples["weak"], 1),
        "strong_truncation_rate": samples["strong_truncated"] / max(samples["strong"], 1),
        "elapsed_s": elapsed,
        "yield_per_paper": n_accepted / max(n_done, 1),
        "rounds_per_paper": n_rounds / max(n_done, 1),
        "usage": meter.totals(),
    })
    logger.info("DONE: papers=%d rounds=%d accepted=%d final=%d outcomes=%s elapsed=%.1fs",
                n_done, n_rounds, n_accepted, n_final, dict(outcomes), elapsed)


if __name__ == "__main__":
    main()

"""Score a solver on the held-out test set: mean@k and best@k (Table 2).

The solver is the weak_solver role, so it answers with the loop's own
solver_generation settings and is graded by the loop's judge. To score a
trained adapter, serve it next to the base model and name it:

    MODEL_DIR=... bash scripts/serve_weak_solver.sh \\
        --enable-lora --lora-modules grpo=<outputs>/<run>/grpo/final
    python scripts/eval_solver.py run_id=eval1 eval.test_set=<data>/test.jsonl \\
        eval.label=grpo models.weak_solver.model=grpo

Writes, under <outputs_root>/<run_id>/eval/<label>:
    per_qa.jsonl    k graded samples per question
    summary.json    mean@k, best@k, questions left out, truncation rate
    usage.jsonl     one line per LLM call
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from autodata.config_runtime import build_client_pool, role_configs_from_dictconfig
from autodata.llm import PromptSet, QAPair, UsageMeter
from autodata.pipeline.at_k import score_test_set, summarize_at_k
from autodata.utils import dump_jsonl, get_logger, load_jsonl, set_seed, setup_logging, write_json

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    load_dotenv(dotenv_path=".secrets/remote.env", override=False)
    set_seed(cfg.seed)
    if not cfg.eval.test_set:
        raise SystemExit("set eval.test_set to a test.jsonl from scripts/build_dataset.py")

    out = Path(cfg.paths.outputs_root) / cfg.run_id / "eval" / str(cfg.eval.label)
    out.mkdir(parents=True, exist_ok=True)
    roles = role_configs_from_dictconfig(cfg.models, cfg.solver_generation)
    meter = UsageMeter(out / "usage.jsonl")
    clients = build_client_pool({r: roles[r] for r in ("weak_solver", "judge")}, meter=meter)

    qas = [QAPair.model_validate(row) for row in load_jsonl(cfg.eval.test_set)]
    k = int(cfg.eval.samples)
    logger.info("scoring %s on %d questions, %d samples each", roles["weak_solver"].model,
                len(qas), k)
    per_qa = score_test_set(qas, clients["weak_solver"], clients["judge"], k,
                            retries=int(cfg.inner_loop.infra_retries),
                            workers=int(cfg.eval.workers), prompts=PromptSet.seed())

    dump_jsonl(out / "per_qa.jsonl", (
        {"paper_id": qa.paper_id, "question": qa.question,
         "samples": [s.model_dump() for s in samples]}
        for qa, samples in zip(qas, per_qa, strict=True)))
    summary = summarize_at_k(per_qa)
    write_json(out / "summary.json", {
        "label": str(cfg.eval.label),
        "model": roles["weak_solver"].model,
        "test_set": str(cfg.eval.test_set),
        "k": k,
        **asdict(summary),
        "usage": meter.totals(),
    })
    logger.info("%s: mean@%d=%s best@%d=%s over %d questions (%d left out)",
                cfg.eval.label, k, summary.mean_at_k, k, summary.best_at_k, summary.n,
                summary.left_out)


if __name__ == "__main__":
    main()

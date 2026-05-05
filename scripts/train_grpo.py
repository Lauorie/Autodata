"""GRPO RL training of Qwen3.5-4B on AutoData synthetic QA pairs.

Reward = weighted rubric score from the judge LLM (over the proxy).
LoRA r=16 on the text-only causal head, BF16, single A800-80GB suffices.

Usage:
    python scripts/train_grpo.py \\
        accepted_qa=/root/autodl-fs/autodata/outputs/full/accepted_qa.jsonl \\
        run_id=grpo1 \\
        train.max_steps=200 train.per_device_batch=2 train.num_generations=4
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from autodata.config_runtime import (
    build_client_pool,
    role_configs_from_dictconfig,
)
from autodata.llm import LLMClient, QAPair
from autodata.pipeline.evaluate_rubric import _ask_judge
from autodata.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


def _load_dataset(jsonl_path: str | Path) -> list[QAPair]:
    import json
    rows: list[QAPair] = []
    with Path(jsonl_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(QAPair.model_validate(json.loads(line)))
    logger.info("loaded %d accepted QAs from %s", len(rows), jsonl_path)
    return rows


def _build_prompt(qa: QAPair) -> str:
    return (
        "PASSAGE:\n\"\"\"\n"
        f"{qa.context}\n"
        "\"\"\"\n\n"
        f"QUESTION: {qa.question}\n\n"
        "ANSWER:"
    )


def _make_reward_fn(qas: list[QAPair], judge: LLMClient):
    """Build a reward function compatible with TRL's GRPOTrainer.

    The dataset row carries a stable ``qa_id`` that TRL forwards to the
    reward function via ``**kwargs`` (TRL >= 0.10 passes through extra
    columns). We look up the rubric/reference from ``qas`` indexed on that
    id rather than the rendered prompt string — prompt strings can be
    truncated, padded, or re-formatted by the trainer and are unsafe keys.
    """
    qa_by_id: dict[str, QAPair] = {f"{q.paper_id}::{q.chunk_id}::{i}": q
                                   for i, q in enumerate(qas)}
    from autodata.pipeline.evaluate_rubric import weighted_average

    def reward_fn(prompts: list[str], completions: list[str], **kwargs):
        qa_ids = kwargs.get("qa_id") or []
        if not qa_ids:
            # No id passthrough means TRL is dropping our extra column —
            # we cannot safely align rewards under shuffle/distributed
            # batching, so refuse rather than scoring the wrong rubric.
            raise RuntimeError(
                "GRPOTrainer did not forward the dataset's `qa_id` column to "
                "reward_fn; cannot compute rewards safely. "
                "Check TRL version (>=0.10) and `remove_unused_columns=False` in GRPOConfig."
            )
        with ThreadPoolExecutor(max_workers=8) as pool:
            def _grade(qid: str, c: str) -> float:
                qa = qa_by_id.get(qid)
                if qa is None:
                    return 0.0
                weights = [cr.weight for cr in qa.rubric.criteria]
                per, _fb = _ask_judge(judge, qa, c)
                return float(weighted_average(per, weights))
            futures = [pool.submit(_grade, qid, c)
                       for qid, c in zip(qa_ids, completions, strict=False)]
            return [f.result() for f in futures]
    return reward_fn


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    load_dotenv(dotenv_path=".secrets/remote.env", override=False)
    set_seed(cfg.seed)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    qa_jsonl = cfg.get("accepted_qa", None)
    if qa_jsonl is None:
        raise SystemExit("must pass accepted_qa=<path/to/accepted_qa.jsonl>")
    qas = _load_dataset(qa_jsonl)
    if not qas:
        raise SystemExit(f"no QA pairs in {qa_jsonl}")

    prompts = [_build_prompt(q) for q in qas]
    qa_ids = [f"{q.paper_id}::{q.chunk_id}::{i}" for i, q in enumerate(qas)]
    ds = Dataset.from_dict({"prompt": prompts, "qa_id": qa_ids})

    role_configs = role_configs_from_dictconfig(cfg.models)
    clients = build_client_pool({k: v for k, v in role_configs.items() if k == "judge"})
    judge = clients["judge"]
    reward_fn = _make_reward_fn(qas, judge)

    model_path = cfg.paths.qwen_model_dir
    out_root = Path(cfg.paths.outputs_root) / cfg.run_id / "grpo"
    out_root.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg.get("train", {})

    grpo_cfg = GRPOConfig(
        output_dir=str(out_root),
        per_device_train_batch_size=int(train_cfg.get("per_device_batch", 2)),
        num_generations=int(train_cfg.get("num_generations", 4)),
        max_prompt_length=int(train_cfg.get("max_prompt_length", 4096)),
        max_completion_length=int(train_cfg.get("max_completion_length", 512)),
        max_steps=int(train_cfg.get("max_steps", 200)),
        learning_rate=float(train_cfg.get("learning_rate", 1e-6)),
        bf16=True,
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 4)),
        save_steps=int(train_cfg.get("save_steps", 50)),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        beta=float(train_cfg.get("beta", 0.04)),
        report_to=[],
        remove_unused_columns=False,    # keep `qa_id` so reward_fn can key on it
    )

    lora_cfg = LoraConfig(
        r=int(train_cfg.get("lora_r", 16)),
        lora_alpha=int(train_cfg.get("lora_alpha", 32)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        # Qwen3.5-4B has mixed full/linear attention layers with different
        # internal projection names. "all-linear" targets every nn.Linear,
        # which is safer than enumerating q_proj/k_proj/v_proj/o_proj
        # (those exist only in full-attention layers).
        target_modules="all-linear",
    )

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    trainer = GRPOTrainer(
        model=model_path,
        args=grpo_cfg,
        train_dataset=ds,
        reward_funcs=reward_fn,
        peft_config=lora_cfg,
        processing_class=tok,
    )
    logger.info("starting GRPO training: model=%s steps=%d bs=%d gens=%d",
                model_path, grpo_cfg.max_steps,
                grpo_cfg.per_device_train_batch_size, grpo_cfg.num_generations)
    trainer.train()
    trainer.save_model(str(out_root / "final"))
    logger.info("GRPO done -> %s/final", out_root)


if __name__ == "__main__":
    main()

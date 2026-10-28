"""GRPO on the solver with the generated questions (arXiv 2606.25996, Sec 3.1.2).

Rollouts take the solver_generation block, the settings the weak solver
answered with in the loop; the reward is the loop's judge on the answer
alone (src/autodata/training/grpo.py). The paper trains Qwen3.5-4B with
batch size 16 and learning rate 1e-6 and reports step 200.

Writes, under <outputs_root>/<run_id>/grpo:
    checkpoints, final/   the trained model (or LoRA adapter)
    judge_usage.jsonl     one line per judge call
    summary.json          settings and judge token totals

Usage (needs the `train` and `serve` extras):
    python scripts/train_grpo.py run_id=grpo1 train.data=<data>/train.jsonl
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from autodata.config_runtime import build_client_pool, role_configs_from_dictconfig
from autodata.llm import PromptSet, QAPair, UsageMeter
from autodata.training import grpo_settings, make_reward_fn, solver_messages
from autodata.utils import get_logger, load_jsonl, set_seed, setup_logging, write_json

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    load_dotenv(dotenv_path=".secrets/remote.env", override=False)
    set_seed(cfg.seed)
    from datasets import Dataset
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    t, gen = cfg.train, dict(cfg.solver_generation)
    if not t.data:
        raise SystemExit("set train.data to a train.jsonl from scripts/build_dataset.py")
    qas = [QAPair.model_validate(row) for row in load_jsonl(t.data)]
    ids = [f"q{i}" for i in range(len(qas))]
    prompts = PromptSet.seed()      # the solver prompt is frozen in every prompt set
    dataset = Dataset.from_dict({"prompt": [solver_messages(prompts, q) for q in qas],
                                 "qa_id": ids})

    out = Path(cfg.paths.outputs_root) / cfg.run_id / "grpo"
    out.mkdir(parents=True, exist_ok=True)
    roles = role_configs_from_dictconfig(cfg.models, cfg.solver_generation)
    meter = UsageMeter(out / "judge_usage.jsonl")
    judge = build_client_pool({"judge": roles["judge"]}, meter=meter)["judge"]
    reward_fn = make_reward_fn(dict(zip(ids, qas, strict=True)), judge, prompts,
                               thinking=bool(gen["thinking"]),
                               retries=int(cfg.inner_loop.infra_retries),
                               workers=int(t.judge_workers))

    settings = grpo_settings(gen, t, world_size=int(os.environ.get("WORLD_SIZE", "1")))
    args = GRPOConfig(
        output_dir=str(out),
        **settings,
        max_steps=int(t.max_steps),
        learning_rate=float(t.learning_rate),
        loss_type=str(t.loss_type),
        beta=float(t.beta),
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=float(t.vllm_gpu_memory_utilization),
        bf16=True,
        gradient_checkpointing=True,
        save_steps=int(t.save_steps),
        logging_steps=int(t.logging_steps),
        report_to=[],
        remove_unused_columns=False,    # reward_fn reads the qa_id column
        seed=int(cfg.seed),
    )
    peft_config = (LoraConfig(r=int(t.lora_r), lora_alpha=2 * int(t.lora_r), bias="none",
                              task_type="CAUSAL_LM", target_modules="all-linear")
                   if t.lora_r else None)
    trainer = GRPOTrainer(model=cfg.paths.qwen_model_dir, reward_funcs=reward_fn, args=args,
                          train_dataset=dataset, peft_config=peft_config)
    logger.info("GRPO: %d questions, %s", len(qas), settings)
    trainer.train()
    trainer.save_model(str(out / "final"))
    write_json(out / "summary.json", {"questions": len(qas), "settings": settings,
                                      "lora_r": t.lora_r, "judge_usage": meter.totals()})
    logger.info("GRPO done -> %s", out / "final")


if __name__ == "__main__":
    main()

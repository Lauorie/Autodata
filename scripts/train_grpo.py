"""GRPO on the solver with the generated questions (arXiv 2606.25996, Sec 3.1.2).

Rollouts take the solver_generation block, the settings the weak solver
answered with in the loop; the reward is the loop's judge on the answer
alone (src/autodata/training/grpo.py). The paper trains Qwen3.5-4B with
batch size 16 and learning rate 1e-6 and reports step 200.

Writes, under <outputs_root>/<run_id>/grpo:
    checkpoints, final/        the trained model (or LoRA adapter)
    judge_usage.rank<N>.jsonl  one line per judge call, one file per process
    summary.json               settings and judge token totals

Full fine-tuning keeps fp32 weights and Adam state (TRL loads fp32 unless
told otherwise; at lr 1e-6 bf16 weights would round most updates away).
That is about 72 GB plus an fp32 reference model, so it is sharded with
ZeRO-3 over two GPUs, each with its own colocated vLLM.

Usage (needs the `train` and `serve` extras):
    accelerate launch --config_file conf/accelerate/zero3.yaml \
        scripts/train_grpo.py run_id=grpo1 train.data=<data>/train.jsonl
"""

from __future__ import annotations

import os
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from autodata.config_runtime import build_client_pool, role_configs_from_dictconfig
from autodata.llm import PromptSet, QAPair, UsageMeter, sum_usage_logs
from autodata.training import grpo_settings, make_reward_fn, solver_messages
from autodata.utils import get_logger, load_jsonl, set_seed, setup_logging, write_json

logger = get_logger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    load_dotenv(dotenv_path=".secrets/remote.env", override=False)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")    # TRL reports runs to the Hub
    set_seed(cfg.seed)
    rank = int(os.environ.get("RANK", "0"))
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
    meter = UsageMeter(out / f"judge_usage.rank{rank}.jsonl")
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
        lr_scheduler_type=str(t.lr_scheduler),
        loss_type=str(t.loss_type),
        beta=float(t.beta),
        bf16=True,
        gradient_checkpointing=True,
        save_steps=int(t.save_steps),
        save_only_model=bool(t.save_only_model),
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
    if trainer.accelerator.is_main_process:     # every rank has finished its judge calls
        write_json(out / "summary.json", {
            "questions": len(qas), "settings": settings, "lora_r": t.lora_r,
            "world_size": trainer.accelerator.num_processes,
            "judge_usage": sum_usage_logs(sorted(out.glob("judge_usage.rank*.jsonl"))),
        })
    logger.info("GRPO done -> %s", out / "final")


if __name__ == "__main__":
    main()

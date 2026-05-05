"""SFT fine-tune Qwen3.5-4B on (question, reference_answer) pairs from AutoData.

Uses the same QAs that AutoData synthesizes (accepted + best-rejected). SFT is
a more pragmatic fit than GRPO for this corpus + this weak model: when the
weak model already nails the questions, GRPO has no reward variance to learn
from, but SFT can still meaningfully push the model's answer style toward the
challenger's reference answers.

Usage:
    python scripts/train_sft.py \\
        accepted_qa=/root/autodl-fs/autodata/outputs/runX/grpo_input.jsonl \\
        run_id=sft1
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig

from autodata.llm.schemas import QAPair
from autodata.utils import set_seed, setup_logging

logger = logging.getLogger(__name__)


def _load_dataset(jsonl_path: str | Path) -> list[QAPair]:
    rows: list[QAPair] = []
    with Path(jsonl_path).open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(QAPair.model_validate(json.loads(line)))
    logger.info("loaded %d QAs from %s", len(rows), jsonl_path)
    return rows


def _format_example(qa: QAPair, tok) -> dict[str, str]:
    """Render a chat-template-formatted SFT example."""
    msgs_full = [
        {"role": "system",
         "content": "You are a research assistant answering a question about a CS paper passage. "
                    "Read the passage carefully, then provide a precise answer that addresses every "
                    "aspect of the question. Reply ONLY with the answer text."},
        {"role": "user",
         "content": f"PASSAGE:\n\"\"\"\n{qa.context}\n\"\"\"\n\nQUESTION: {qa.question}\n\nANSWER:"},
        {"role": "assistant", "content": qa.reference_answer},
    ]
    text = tok.apply_chat_template(msgs_full, tokenize=False, add_generation_prompt=False)
    return {"text": text}


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    setup_logging("INFO")
    load_dotenv(dotenv_path=".secrets/remote.env", override=False)
    set_seed(cfg.seed)

    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    qa_jsonl = cfg.get("accepted_qa", None)
    if qa_jsonl is None:
        raise SystemExit("must pass accepted_qa=<path/to/qa.jsonl>")
    qas = _load_dataset(qa_jsonl)
    if not qas:
        raise SystemExit(f"no QA pairs in {qa_jsonl}")

    model_path = cfg.paths.qwen_model_dir
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    examples = [_format_example(q, tok) for q in qas]
    ds = Dataset.from_list(examples)
    logger.info("dataset: %d rows, sample length stats: %s",
                len(ds), {"min": min(len(e["text"]) for e in examples),
                          "max": max(len(e["text"]) for e in examples),
                          "mean": int(sum(len(e["text"]) for e in examples) / len(examples))})

    out_root = Path(cfg.paths.outputs_root) / cfg.run_id / "sft"
    out_root.mkdir(parents=True, exist_ok=True)
    train_cfg = cfg.get("train", {})

    sft_cfg = SFTConfig(
        output_dir=str(out_root),
        per_device_train_batch_size=int(train_cfg.get("per_device_batch", 1)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 4)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),  # higher LR for LoRA
        max_steps=int(train_cfg.get("max_steps", 50)),
        max_length=int(train_cfg.get("max_length", 4096)),
        bf16=True,
        save_steps=int(train_cfg.get("save_steps", 50)),
        logging_steps=int(train_cfg.get("logging_steps", 2)),
        warmup_ratio=0.05,
        report_to=[],
        remove_unused_columns=False,
    )

    lora_cfg = LoraConfig(
        r=int(train_cfg.get("lora_r", 16)),
        lora_alpha=int(train_cfg.get("lora_alpha", 32)),
        lora_dropout=float(train_cfg.get("lora_dropout", 0.05)),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )

    trainer = SFTTrainer(
        model=model_path,
        args=sft_cfg,
        train_dataset=ds,
        peft_config=lora_cfg,
        processing_class=tok,
    )
    logger.info("starting SFT: max_steps=%d bs=%d grad_accum=%d lr=%g",
                sft_cfg.max_steps, sft_cfg.per_device_train_batch_size,
                sft_cfg.gradient_accumulation_steps, sft_cfg.learning_rate)
    trainer.train()
    trainer.save_model(str(out_root / "final"))
    logger.info("SFT done -> %s/final", out_root)


if __name__ == "__main__":
    main()

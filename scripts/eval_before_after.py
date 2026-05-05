"""Compare base Qwen3.5-4B vs the GRPO-finetuned LoRA on the held-out QAs.

For each held-out QA pair:
  1. Generate an answer with the base model.
  2. Generate an answer with the LoRA-merged model.
  3. Score both with the judge (same judge as during training).
  4. Report mean weighted score before vs after, plus a per-example diff.

Usage:
    python scripts/eval_before_after.py \\
        --base /root/autodl-fs/autodata/Qwen3.5-4B \\
        --lora /root/autodl-fs/autodata/outputs/run20b/grpo/final \\
        --eval-jsonl /root/autodl-fs/autodata/outputs/run20b/grpo_input.jsonl \\
        --max-eval 10 \\
        --models-yaml conf/models/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)


def _load_qas(path: Path, n: int) -> list:
    from autodata.llm.schemas import QAPair
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(QAPair.model_validate(json.loads(line)))
        if len(rows) >= n:
            break
    return rows


def _generate(model, tokenizer, qa, max_new_tokens: int = 384) -> str:
    import torch
    msgs = [
        {"role": "system",
         "content": "You are a research assistant answering a question about a CS paper passage. "
                    "Read the passage carefully, then provide a precise answer that addresses every "
                    "aspect of the question. Reply ONLY with the answer text."},
        {"role": "user",
         "content": f"PASSAGE:\n\"\"\"\n{qa.context}\n\"\"\"\n\nQUESTION: {qa.question}\n\nANSWER:"},
    ]
    prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
    gen = out[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True, help="Path to GRPO output dir containing adapter_config.json")
    ap.add_argument("--eval-jsonl", required=True)
    ap.add_argument("--max-eval", type=int, default=10)
    ap.add_argument("--models-yaml", default="conf/models/default.yaml")
    args = ap.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from autodata.config_runtime import build_role_clients_from_yaml
    from autodata.pipeline.evaluate_rubric import _ask_judge, weighted_average

    qas = _load_qas(Path(args.eval_jsonl), args.max_eval)
    logger.info("evaluating on %d QAs", len(qas))

    judge = build_role_clients_from_yaml(args.models_yaml)["judge"]

    logger.info("loading base model %s", args.base)
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                trust_remote_code=True, device_map="cuda:0")
    base.eval()

    rows = []
    base_scores: list[float] = []
    logger.info("scoring base model...")
    for i, qa in enumerate(qas):
        ans = _generate(base, tok, qa)
        weights = [c.weight for c in qa.rubric.criteria]
        per, _fb = _ask_judge(judge, qa, ans)
        score = float(weighted_average(per, weights))
        base_scores.append(score)
        rows.append({"i": i, "q": qa.question[:120], "base_answer": ans[:300],
                     "base_score": score})
        logger.info("  base %d/%d -> %.3f", i + 1, len(qas), score)

    # Free base model's memory before loading LoRA
    del base
    torch.cuda.empty_cache()

    logger.info("loading LoRA from %s", args.lora)
    base2 = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16,
                                                 trust_remote_code=True, device_map="cuda:0")
    finetuned = PeftModel.from_pretrained(base2, args.lora)
    finetuned.eval()

    logger.info("scoring fine-tuned model...")
    ft_scores: list[float] = []
    for i, qa in enumerate(qas):
        ans = _generate(finetuned, tok, qa)
        weights = [c.weight for c in qa.rubric.criteria]
        per, _fb = _ask_judge(judge, qa, ans)
        score = float(weighted_average(per, weights))
        ft_scores.append(score)
        rows[i]["ft_answer"] = ans[:300]
        rows[i]["ft_score"] = score
        rows[i]["delta"] = score - rows[i]["base_score"]
        logger.info("  ft   %d/%d -> %.3f (delta %+.3f)", i + 1, len(qas), score,
                    rows[i]["delta"])

    base_mean = sum(base_scores) / len(base_scores)
    ft_mean = sum(ft_scores) / len(ft_scores)
    logger.info("=" * 60)
    logger.info("base_mean=%.3f  ft_mean=%.3f  delta=%+.3f", base_mean, ft_mean, ft_mean - base_mean)
    n_better = sum(1 for r in rows if r["delta"] > 0)
    n_worse = sum(1 for r in rows if r["delta"] < 0)
    logger.info("per-example: %d better, %d worse, %d tied", n_better, n_worse,
                len(rows) - n_better - n_worse)

    out = Path(args.eval_jsonl).parent / "before_after_eval.json"
    out.write_text(json.dumps({
        "base_mean": base_mean, "ft_mean": ft_mean, "delta": ft_mean - base_mean,
        "n_eval": len(qas), "n_better": n_better, "n_worse": n_worse,
        "rows": rows,
    }, indent=2, ensure_ascii=False))
    logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

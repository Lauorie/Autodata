"""GRPO pieces that do not need TRL (arXiv 2606.25996, Sec 3.1.2).

The loop accepted each question because the weak solver, answering with the
``solver_generation`` settings, scored low on it. GRPO rollouts use those
same settings: the solver system prompt under the chat template, the
thinking switch, the token budget and the sampling. The reward is the
loop's judge on the answer alone, as vLLM's reasoning parser hands it over
in the loop:

- text after the last ``</think>`` is the answer;
- with thinking on, no ``</think>`` means the budget ran out mid-thought:
  no answer, reward 0, as the loop scores a truncated empty answer;
- a judge that never returns usable scores, or a gateway that stays down,
  makes the completion unscorable (None). TRL then leaves it out of the
  group baseline and gives it no advantage, instead of charging the model
  for an infrastructure failure.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..llm import LLMClient, QAPair
from ..llm.errors import LLMError
from ..llm.prompt_set import PromptSet
from ..pipeline.evaluate_rubric import JudgeOutputError, judge_answer, rubric_score

logger = logging.getLogger(__name__)

# Keys a response parser may split the thought into.
_REASONING_KEYS = ("reasoning_content", "reasoning", "thinking")


def solver_messages(prompts: PromptSet, qa: QAPair) -> list[dict[str, str]]:
    """The weak solver's chat, as the loop sends it."""
    return [{"role": "system", "content": prompts.solver_system},
            {"role": "user", "content": prompts.render_solver(qa.context, qa.question)}]


def answer_text(completion: str | list[dict[str, Any]], thinking: bool) -> str:
    """The part of a completion the judge grades."""
    if isinstance(completion, str):
        text = completion
    else:
        msg = completion[-1]
        text = msg.get("content") or ""
        if any(k in msg for k in _REASONING_KEYS):   # already split by a parser
            return text.rsplit("</think>", 1)[-1].strip()
    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if thinking or "<think>" in text:
        return ""
    return text.strip()


def make_reward_fn(qas_by_id: Mapping[str, QAPair], judge: LLMClient, prompts: PromptSet,
                   thinking: bool, retries: int, workers: int) -> Callable[..., list[float | None]]:
    """A TRL reward function keyed on the dataset's ``qa_id`` column."""
    def grade(qid: str, completion: Any) -> float | None:
        qa = qas_by_id[qid]
        answer = answer_text(completion, thinking)
        if not answer:
            return 0.0
        try:
            per, _feedback = judge_answer(judge, qa, answer, retries, prompts=prompts)
        except (JudgeOutputError, LLMError) as e:
            logger.warning("unscorable completion for %s: %s", qid, e)
            return None
        return rubric_score(per, qa.rubric.weights)

    def reward_fn(completions: list[Any], qa_id: list[str] | None = None,
                  **_kwargs: Any) -> list[float | None]:
        if not qa_id:
            raise RuntimeError("the trainer did not pass the dataset's qa_id column; set "
                               "remove_unused_columns=False")
        if missing := sorted(set(qa_id) - set(qas_by_id)):
            raise KeyError(f"unknown qa_id: {missing[:3]}")
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            return list(pool.map(grade, qa_id, completions))

    return reward_fn


def grpo_settings(gen: Mapping[str, Any], train: Mapping[str, Any],
                  world_size: int) -> dict[str, Any]:
    """GRPOConfig fields fixed by ``solver_generation`` and the batch shape.

    The batch is counted in prompts per optimizer step; gradient
    accumulation makes up the difference.
    """
    presence = gen.get("presence_penalty")
    if presence is not None and not train["use_vllm"]:
        raise ValueError("presence_penalty needs use_vllm: transformers generation has none, "
                         "and dropping it would change the solver's sampling")
    completions = int(train["prompts_per_step"]) * int(train["num_generations"])
    per_pass = int(train["per_device_batch"]) * world_size
    if completions % per_pass:
        raise ValueError(f"prompts_per_step x num_generations = {completions} is not a "
                         f"multiple of per_device_batch x world_size = {per_pass}")
    settings: dict[str, Any] = {
        "per_device_train_batch_size": int(train["per_device_batch"]),
        "gradient_accumulation_steps": completions // per_pass,
        "num_generations": int(train["num_generations"]),
        "max_completion_length": int(gen["max_tokens"]),
        "temperature": float(gen["temperature"]),
        "top_p": float(gen["top_p"]) if gen.get("top_p") is not None else 1.0,
        "top_k": int(gen["top_k"]) if gen.get("top_k") is not None else 0,
        "chat_template_kwargs": {"enable_thinking": bool(gen["thinking"])},
        "use_vllm": bool(train["use_vllm"]),
    }
    if presence is not None:
        settings["generation_kwargs"] = {"presence_penalty": float(presence)}
    if train["use_vllm"]:
        settings["vllm_max_model_length"] = int(gen["max_tokens"]) + int(train["max_prompt_tokens"])
    return settings


__all__ = ["answer_text", "grpo_settings", "make_reward_fn", "solver_messages"]

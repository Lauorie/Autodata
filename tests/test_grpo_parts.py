"""GRPO pieces that do not need TRL (arXiv 2606.25996, Sec 3.1.2).

The rollouts must match the loop's weak solver: same system prompt, chat
template, thinking switch, token budget and sampling. The reward grades the
answer, not the thought, with the loop's judge; a judge failure is
unscorable (None), not a zero.
"""

from __future__ import annotations

from typing import Any

import pytest

from autodata.llm.prompt_set import PromptSet
from autodata.training.grpo import answer_text, grpo_settings, make_reward_fn, solver_messages
from tests.fakes import WEAK_PER, ScriptedBackend, judge_payload, make_clients, make_qa

SEED = PromptSet.seed()
GEN = {"thinking": True, "max_tokens": 65536, "temperature": 1.0, "top_p": 0.95, "top_k": 20,
       "presence_penalty": 1.5}
TRAIN = {"prompts_per_step": 16, "num_generations": 8, "per_device_batch": 8,
         "use_vllm": True, "max_prompt_tokens": 8192, "vllm_gpu_memory_utilization": 0.35,
         "vllm_sleep": True, "vllm_is_mode": "token_truncate"}


def test_prompt_is_the_solver_chat() -> None:
    qa = make_qa()
    assert solver_messages(SEED, qa) == [
        {"role": "system", "content": SEED.solver_system},
        {"role": "user", "content": SEED.render_solver(qa.context, qa.question)},
    ]


# -- what the judge sees ---------------------------------------------------------

@pytest.mark.parametrize(("completion", "thinking", "expected"), [
    ("weighing options...\n</think>\n\nThe answer.", True, "The answer."),
    ("<think>a</think>b</think> final", True, "final"),
    ("still thinking when the budget ran out", True, ""),
    ("<think>never closed", False, ""),
    ("plain answer", False, "plain answer"),
    ([{"role": "assistant", "content": "x</think>Chat answer."}], True, "Chat answer."),
    ([{"role": "assistant", "reasoning_content": "thought", "content": "Parsed."}], True,
     "Parsed."),
])
def test_answer_text_drops_the_thought(completion: Any, thinking: bool, expected: str) -> None:
    assert answer_text(completion, thinking) == expected


# -- the reward ------------------------------------------------------------------

def _reward(judge: Any) -> tuple[Any, ScriptedBackend]:
    backend = ScriptedBackend(handlers={"judge": judge})
    qa = make_qa()
    fn = make_reward_fn({"id0": qa}, make_clients(backend)["judge"], SEED, thinking=True,
                        retries=0, workers=1)
    return fn, backend


def test_reward_grades_the_answer_only() -> None:
    fn, backend = _reward(lambda m: judge_payload(WEAK_PER))
    out = fn(prompts=["p"], completions=["SECRET-THOUGHT</think>THE-ANSWER"], qa_id=["id0"])
    assert out == [pytest.approx(0.5)]
    seen = backend.text_seen_by("judge")
    assert "THE-ANSWER" in seen and "SECRET-THOUGHT" not in seen


def test_no_answer_scores_zero_without_a_judge_call() -> None:
    fn, backend = _reward(lambda m: judge_payload(WEAK_PER))
    assert fn(prompts=["p"], completions=["cut off mid-thought"], qa_id=["id0"]) == [0.0]
    assert backend.calls == []


def test_judge_failure_is_unscorable() -> None:
    fn, _ = _reward(lambda m: "not json")
    assert fn(prompts=["p"], completions=["t</think>answer"], qa_id=["id0"]) == [None]


def test_reward_needs_the_question_id() -> None:
    fn, _ = _reward(lambda m: judge_payload(WEAK_PER))
    with pytest.raises(RuntimeError, match="qa_id"):
        fn(prompts=["p"], completions=["t</think>a"])
    with pytest.raises(KeyError):
        fn(prompts=["p"], completions=["t</think>a"], qa_id=["unknown"])


# -- the trainer settings ----------------------------------------------------------

def test_rollouts_share_the_budget_and_thinking() -> None:
    s = grpo_settings(GEN, TRAIN, world_size=1)
    assert s["max_completion_length"] == 65536
    assert s["temperature"] == 1.0
    assert s["chat_template_kwargs"] == {"enable_thinking": True}
    assert s["vllm_max_model_length"] == 65536 + 8192


def test_rollouts_sample_the_policy_itself() -> None:
    # vLLM reports processed logprobs: a presence penalty or top-k/top-p cut
    # would move them off the trainer's logits/T and the importance ratio
    # would mask every long completion. Those stay decode-time choices for
    # the loop and the held-out eval.
    s = grpo_settings(GEN, TRAIN, world_size=1)
    assert not {"top_p", "top_k", "generation_kwargs"} & s.keys()


def test_importance_sampling_mode_follows_the_config() -> None:
    s = grpo_settings(GEN, {**TRAIN, "vllm_is_mode": "token_truncate"}, world_size=1)
    assert s["vllm_importance_sampling_mode"] == "token_truncate"


def test_batch_is_counted_in_prompts() -> None:
    s = grpo_settings(GEN, TRAIN, world_size=1)
    assert s["per_device_train_batch_size"] * s["gradient_accumulation_steps"] == 16 * 8
    assert s["num_generations"] == 8


def test_colocated_vllm_follows_the_config() -> None:
    s = grpo_settings(GEN, {**TRAIN, "vllm_gpu_memory_utilization": 0.3, "vllm_sleep": False},
                      world_size=2)
    assert s["vllm_mode"] == "colocate"
    assert (s["vllm_gpu_memory_utilization"], s["vllm_enable_sleep_mode"]) == (0.3, False)


def test_uneven_batch_is_refused() -> None:
    with pytest.raises(ValueError, match="prompts_per_step"):
        grpo_settings(GEN, {**TRAIN, "per_device_batch": 7}, world_size=1)

"""End-to-end inner-loop test using fake LLM clients (no network).

Validates that:
- A failing QV makes the loop iterate with QV feedback.
- A weak/strong gap that crosses thresholds yields ACCEPTED.
- Without an acceptable round, the loop terminates after max_iterations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from autodata.data_module.chunker import Chunk
from autodata.llm import LLMClient
from autodata.pipeline.acceptance import AcceptanceCriteria
from autodata.pipeline.inner_loop import InnerLoopConfig, run_inner_loop_for_chunk


@dataclass
class _FakeBackend:
    """Returns canned responses keyed on which role is calling."""
    handlers: dict[str, list[str]]

    def chat(self, role_cfg, messages):
        h = self.handlers.get(role_cfg.role, [])
        if not h:
            raise RuntimeError(f"no handler left for role {role_cfg.role}")
        return h.pop(0)


def _client(role: str, backend) -> LLMClient:
    from autodata.llm.client import LLMRoleConfig
    return LLMClient(LLMRoleConfig(role=role, endpoint="proxy", model="fake"), backend)


def _qa_json() -> str:
    return json.dumps({
        "question": "Why does X imply Y in this paper's setup?",
        "reference_answer": "Because of the assumption stated in section 3.",
        "rubric": {
            "criteria": [
                {"description": "answer mentions assumption from section 3", "weight": 4},
                {"description": "answer makes the implication X -> Y explicit", "weight": 3},
            ]
        }
    })


def _judge_json(per: list[float]) -> str:
    return json.dumps({"per_criterion": per, "feedback": "ok"})


def _make_chunk() -> Chunk:
    return Chunk(
        paper_id="papX", title="Paper X", section="Method",
        text="Section 3 assumes A. Therefore X implies Y. " * 30,
        chunk_id="papX::01::00",
    )


def _common_inner_cfg() -> InnerLoopConfig:
    return InnerLoopConfig(max_iterations=3, num_solver_samples=2,
                           qa_per_chunk=1, parallel_solvers=False)


def test_acceptance_path() -> None:
    # Round 0: QV passes, weak fails, strong succeeds, gap big -> ACCEPTED.
    backend = _FakeBackend(handlers={
        "challenger": [_qa_json()],
        "quality_verifier": [json.dumps({"passed": True, "feedback": "ok", "issues": []})],
        # 2 weak samples * (no judge for solver text) followed by 2 judge calls
        "weak_solver": ["weak ans 1", "weak ans 2"],
        "strong_solver": ["strong ans 1", "strong ans 2"],
        # JUDGE order: weak1, weak2, strong1, strong2 (parallel pool may shuffle within each pair,
        # but the *pair* of weak judge calls fires before the strong pair starts).
        "judge": [
            _judge_json([0.3, 0.3]),  # weak1 -> 0.3*4+0.3*3 / 7 = 0.30
            _judge_json([0.3, 0.4]),  # weak2 -> 0.34
            _judge_json([0.9, 0.9]),  # strong1 -> 0.90
            _judge_json([0.95, 0.85]),  # strong2 -> ~0.91
        ],
    })
    clients = {
        "challenger": _client("challenger", backend),
        "quality_verifier": _client("quality_verifier", backend),
        "weak_solver": _client("weak_solver", backend),
        "strong_solver": _client("strong_solver", backend),
        "judge": _client("judge", backend),
    }
    qa, rounds = run_inner_loop_for_chunk(
        _make_chunk(), clients, AcceptanceCriteria(), _common_inner_cfg(),
    )
    assert qa is not None
    assert len(rounds) == 1
    assert rounds[0].accepted is True


def test_qv_failure_iterates() -> None:
    backend = _FakeBackend(handlers={
        "challenger": [_qa_json(), _qa_json()],
        "quality_verifier": [
            json.dumps({"passed": False, "feedback": "context leak", "issues": ["context_leak"]}),
            json.dumps({"passed": True, "feedback": "ok", "issues": []}),
        ],
        "weak_solver": ["weak1", "weak2"],
        "strong_solver": ["strong1", "strong2"],
        "judge": [
            _judge_json([0.3, 0.3]),
            _judge_json([0.3, 0.3]),
            _judge_json([0.9, 0.9]),
            _judge_json([0.9, 0.9]),
        ],
    })
    clients = {
        "challenger": _client("challenger", backend),
        "quality_verifier": _client("quality_verifier", backend),
        "weak_solver": _client("weak_solver", backend),
        "strong_solver": _client("strong_solver", backend),
        "judge": _client("judge", backend),
    }
    qa, rounds = run_inner_loop_for_chunk(
        _make_chunk(), clients, AcceptanceCriteria(), _common_inner_cfg(),
    )
    assert qa is not None
    # Round 0 was QV-rejected, round 1 accepted.
    assert len(rounds) == 2
    assert rounds[0].accepted is False
    assert rounds[1].accepted is True


def test_exhausts_iterations() -> None:
    # Strong always too low -> never accepted.
    n_iters = 2
    backend = _FakeBackend(handlers={
        "challenger": [_qa_json()] * n_iters,
        "quality_verifier": [json.dumps({"passed": True, "feedback": "", "issues": []})] * n_iters,
        "weak_solver": ["w"] * (2 * n_iters),
        "strong_solver": ["s"] * (2 * n_iters),
        "judge": ([_judge_json([0.3, 0.3])] * 2 + [_judge_json([0.4, 0.4])] * 2) * n_iters,
    })
    clients = {
        "challenger": _client("challenger", backend),
        "quality_verifier": _client("quality_verifier", backend),
        "weak_solver": _client("weak_solver", backend),
        "strong_solver": _client("strong_solver", backend),
        "judge": _client("judge", backend),
    }
    qa, rounds = run_inner_loop_for_chunk(
        _make_chunk(), clients, AcceptanceCriteria(),
        InnerLoopConfig(max_iterations=n_iters, num_solver_samples=2,
                        qa_per_chunk=1, parallel_solvers=False),
    )
    assert qa is None
    assert len(rounds) == n_iters
    assert all(not r.accepted for r in rounds)

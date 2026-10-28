"""Shipped role configs and the shared solver generation block.

Both solvers take thinking, budget, and sampling from one ``solver_generation``
block, on both loading paths (Hydra for the scripts, plain YAML for CLIs).
"""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from autodata.config_runtime import role_configs_from_dictconfig, role_configs_from_yaml
from autodata.llm import LLMRoleConfig

CONF = Path(__file__).resolve().parents[1] / "conf"
SOLVERS = ("weak_solver", "strong_solver")


def _from_hydra() -> dict[str, LLMRoleConfig]:
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose(config_name="config", overrides=["run_id=t"])
    return role_configs_from_dictconfig(cfg.models, cfg.solver_generation)


def _from_yaml() -> dict[str, LLMRoleConfig]:
    return role_configs_from_yaml(CONF / "models" / "default.yaml", CONF / "config.yaml")


@pytest.mark.parametrize("load", [_from_hydra, _from_yaml])
def test_solvers_share_the_generation_block(load) -> None:
    roles = load()
    weak, strong = roles["weak_solver"], roles["strong_solver"]
    for field in ("thinking", "max_tokens", "temperature", "top_p", "top_k", "presence_penalty"):
        assert getattr(weak, field) == getattr(strong, field), field
    assert weak.thinking is True
    assert weak.temperature == 1.0            # paper Sec 4: solvers at temperature 1.0
    assert weak.max_tokens == 65536           # paper App A reasoning budget


@pytest.mark.parametrize("load", [_from_hydra, _from_yaml])
def test_verifier_and_judge_do_not_think(load) -> None:
    roles = load()
    assert roles["challenger"].thinking is True
    assert roles["quality_verifier"].thinking is False
    assert roles["judge"].thinking is False


def test_weak_solver_is_served_by_vllm() -> None:
    weak = _from_yaml()["weak_solver"]
    assert weak.endpoint == "vllm"
    assert not weak.model.startswith("/")     # a served model name, not a host path

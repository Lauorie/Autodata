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


# Kimi-K3 behind the PaperBypass gateway, probed 2026-10-28: it thinks whatever
# the request says, and it rejects any temperature but 0.0, 0.6 and 1.0 (0.7
# came back as a 502 "upstream error 400").
_KIMI_K3_TEMPERATURES = {0.0, 0.6, 1.0}
_KIMI_ROLES = ("challenger", "quality_verifier", "judge", "meta_optimizer")


@pytest.mark.parametrize("load", [_from_hydra, _from_yaml])
def test_kimi_roles_match_what_the_gateway_accepts(load) -> None:
    roles = load()
    for name in _KIMI_ROLES:
        role = roles[name]
        assert role.model == "moonshotai/kimi-k3", name
        assert role.thinking is True, name           # budget max_tokens for the thought
        assert role.temperature in _KIMI_K3_TEMPERATURES, name
    assert roles["quality_verifier"].temperature == 0.0
    assert roles["judge"].temperature == 0.0
    assert roles["judge"].max_tokens >= 16384        # thought + one score per rubric item


def test_strong_solver_is_qwen38_max() -> None:
    assert _from_yaml()["strong_solver"].model == "qwen/qwen3.8-max"


def test_weak_solver_is_served_by_vllm() -> None:
    weak = _from_yaml()["weak_solver"]
    assert weak.endpoint == "vllm"
    assert not weak.model.startswith("/")     # a served model name, not a host path


# Flash profile (models=flash), for fast pipeline runs. DeepSeek-V4.1-Flash
# through the gateway, probed 2026-10-28: thinks even with effort "none";
# accepts temperature 0.0, 0.7 and 1.0.
_DEEPSEEK_FLASH_TEMPERATURES = {0.0, 0.7, 1.0}


def _flash_from_hydra() -> dict[str, LLMRoleConfig]:
    with initialize_config_dir(config_dir=str(CONF), version_base=None):
        cfg = compose(config_name="config", overrides=["run_id=t", "models=flash"])
    return role_configs_from_dictconfig(cfg.models, cfg.solver_generation)


def test_flash_profile_matches_what_the_gateway_accepts() -> None:
    roles = _flash_from_hydra()
    for name in _KIMI_ROLES:                      # the same four LLM-only roles
        role = roles[name]
        assert role.model == "deepseek/deepseek-v4.1-flash", name
        assert role.thinking is True, name
        assert role.temperature in _DEEPSEEK_FLASH_TEMPERATURES, name
    assert roles["judge"].temperature == 0.0 and roles["judge"].max_tokens >= 16384
    assert roles["strong_solver"].model == "qwen/qwen3.8-flash"
    assert roles["weak_solver"].endpoint == "vllm"


def test_flash_solvers_still_share_the_generation_block() -> None:
    roles = _flash_from_hydra()
    weak, strong = roles["weak_solver"], roles["strong_solver"]
    for field in ("thinking", "max_tokens", "temperature", "top_p", "top_k", "presence_penalty"):
        assert getattr(weak, field) == getattr(strong, field), field

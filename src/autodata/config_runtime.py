"""Bridge between Hydra YAML configs and the runtime LLM client pool.

Kept separate from the dataclass-only `LLMRoleConfig` so that this module
can pull in heavy deps (hydra, yaml) without making the schemas depend on them.

Both solvers take thinking, budget, and sampling from the ``solver_generation``
block of conf/config.yaml, applied here in code on every loading path.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .llm import LLMClient
from .llm.client import LLMRoleConfig, build_client_pool

logger = logging.getLogger(__name__)

SOLVER_ROLES = ("weak_solver", "strong_solver")


def _optional(body: Mapping[str, Any], key: str, cast: type) -> Any:
    value = body.get(key)
    return None if value is None else cast(value)


def _build_role(role: str, body: Mapping[str, Any]) -> LLMRoleConfig:
    return LLMRoleConfig(
        role=role,
        endpoint=body["endpoint"],
        model=body["model"],
        temperature=float(body.get("temperature", 0.7)),
        max_tokens=int(body.get("max_tokens", 2000)),
        thinking=bool(body.get("thinking", False)),
        reasoning_effort=str(body.get("reasoning_effort", "auto")),
        top_p=_optional(body, "top_p", float),
        top_k=_optional(body, "top_k", int),
        presence_penalty=_optional(body, "presence_penalty", float),
        base_url_env=str(body.get("base_url_env", "PROXY_BASE_URL")),
        api_key_env=str(body.get("api_key_env", "PROXY_API_KEY")),
    )


def apply_solver_generation(roles: dict[str, LLMRoleConfig],
                            gen: Mapping[str, Any]) -> dict[str, LLMRoleConfig]:
    """Give both solvers the same thinking switch, budget, and sampling."""
    fields = {
        "thinking": bool(gen["thinking"]),
        "max_tokens": int(gen["max_tokens"]),
        "temperature": float(gen["temperature"]),
        "top_p": _optional(gen, "top_p", float),
        "top_k": _optional(gen, "top_k", int),
        "presence_penalty": _optional(gen, "presence_penalty", float),
    }
    return {name: replace(cfg, **fields) if name in SOLVER_ROLES else cfg
            for name, cfg in roles.items()}


def role_configs_from_yaml(models_path: str | Path,
                           config_path: str | Path) -> dict[str, LLMRoleConfig]:
    raw = yaml.safe_load(Path(models_path).read_text())
    gen = yaml.safe_load(Path(config_path).read_text())["solver_generation"]
    roles = {role: _build_role(role, body) for role, body in raw.items() if isinstance(body, dict)}
    return apply_solver_generation(roles, gen)


def role_configs_from_dictconfig(models: Any, solver_generation: Any) -> dict[str, LLMRoleConfig]:
    """Convert a Hydra/OmegaConf models block into role configs."""
    items = models.items() if hasattr(models, "items") else dict(models).items()
    roles = {role: _build_role(role, dict(body)) for role, body in items}
    return apply_solver_generation(roles, dict(solver_generation))


def build_role_clients_from_yaml(models_path: str | Path,
                                 config_path: str | Path) -> dict[str, LLMClient]:
    return build_client_pool(role_configs_from_yaml(models_path, config_path))


__all__ = [
    "SOLVER_ROLES",
    "apply_solver_generation",
    "build_role_clients_from_yaml",
    "role_configs_from_dictconfig",
    "role_configs_from_yaml",
]

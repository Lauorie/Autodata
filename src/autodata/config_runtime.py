"""Bridge between Hydra YAML configs and the runtime LLM client pool.

Kept separate from the dataclass-only `LLMRoleConfig` so that this module
can pull in heavy deps (hydra, yaml) without making the schemas depend on them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from .llm import LLMClient
from .llm.client import LLMRoleConfig, build_client_pool

logger = logging.getLogger(__name__)


def _build_role(role: str, body: dict[str, Any]) -> LLMRoleConfig:
    return LLMRoleConfig(
        role=role,
        endpoint=body["endpoint"],
        model=body["model"],
        temperature=float(body.get("temperature", 0.7)),
        max_tokens=int(body.get("max_tokens", 2000)),
        reasoning=bool(body.get("reasoning", False)),
        reasoning_effort=str(body.get("reasoning_effort", "auto")),
    )


def role_configs_from_yaml(path: str | Path) -> dict[str, LLMRoleConfig]:
    raw = yaml.safe_load(Path(path).read_text())
    return {role: _build_role(role, body) for role, body in raw.items() if isinstance(body, dict)}


def role_configs_from_dictconfig(cfg: Any) -> dict[str, LLMRoleConfig]:
    """Convert a Hydra/OmegaConf models block into role configs."""
    items = cfg.items() if hasattr(cfg, "items") else dict(cfg).items()
    return {role: _build_role(role, dict(body)) for role, body in items}


def build_role_clients_from_yaml(path: str | Path) -> dict[str, LLMClient]:
    return build_client_pool(role_configs_from_yaml(path))


__all__ = [
    "build_role_clients_from_yaml",
    "role_configs_from_dictconfig",
    "role_configs_from_yaml",
]

"""
Central configuration loader.

Reads `configs/default.yaml` once and exposes it as a nested attribute-accessible
object. All other modules import `get_config()` from here; nothing else reads
YAML directly.

Usage:
    from src.utils.config import get_config

    cfg = get_config()
    print(cfg.dataset.sampling_rate)   # 500
    print(cfg.paths.data_raw)          # absolute Path to data/raw/ptb-xl
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Repository root is two levels up from this file: src/utils/config.py -> repo root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"


class ConfigNode(dict):
    """
    Dict subclass that also supports attribute access.

    Lets you write `cfg.dataset.sampling_rate` instead of
    `cfg["dataset"]["sampling_rate"]`. Recursively wraps nested dicts.
    """

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(f"Config has no key '{name}'") from exc
        # Recursively wrap nested dicts so `cfg.dataset.sampling_rate` works too
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            self[name] = value
        return value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def _resolve_paths(node: Any, project_root: Path) -> Any:
    """
    Walk the config tree and convert any path-like strings under the 'paths'
    section into absolute Path objects.

    This means code can do `cfg.paths.data_raw` and get a real Path, without
    needing to know where the project lives on disk.
    """
    if isinstance(node, dict):
        return {k: _resolve_paths(v, project_root) for k, v in node.items()}
    if isinstance(node, list):
        return [_resolve_paths(v, project_root) for v in node]
    if isinstance(node, str) and node.startswith(("data/", "models/", "logs/", "./")):
        return (project_root / node).resolve()
    return node


@lru_cache(maxsize=1)
def get_config(config_path: Path | None = None) -> ConfigNode:
    """
    Load config from YAML. Cached after first call (edit-and-reload requires
    restarting Python, which is fine for training runs).

    Args:
        config_path: Optional override. Defaults to configs/default.yaml.

    Returns:
        ConfigNode with attribute access and resolved absolute paths.
    """
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping, got {type(raw).__name__}")

    # Resolve paths in the top-level 'paths' section only
    if "paths" in raw:
        raw["paths"] = _resolve_paths(raw["paths"], PROJECT_ROOT)

    return ConfigNode(raw)


if __name__ == "__main__":
    # Quick smoke test: `python -m src.utils.config`
    cfg = get_config()
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("Config path  :", DEFAULT_CONFIG_PATH)
    print()
    print("Dataset      :", cfg.dataset.name, cfg.dataset.version)
    print("Sampling rate:", cfg.dataset.sampling_rate, "Hz")
    print("Signal length:", cfg.dataset.signal_length)
    print("Superclasses :", cfg.labels.superclasses)
    print()
    print("Raw data dir :", cfg.paths.data_raw)
    print("Processed dir:", cfg.paths.data_processed)
    print("Data raw exists:", cfg.paths.data_raw.exists())

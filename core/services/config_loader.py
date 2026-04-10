"""Configuration loading and validation for morphogenesis-ga experiments."""
from __future__ import annotations

from typing import Any, Dict

import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file and return as dict."""
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(cfg: Dict[str, Any]) -> None:
    """Validate required config sections exist. Raises ValueError on missing."""
    required_sections = ["world", "target", "model", "ga", "fitness"]
    missing = [s for s in required_sections if s not in cfg]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")

    # Validate value ranges
    ga = cfg.get("ga", {})
    if ga.get("population_size", 1) < 4:
        raise ValueError("ga.population_size must be >= 4")
    elite_frac = ga.get("elite_fraction", 0.25)
    if not (0 < elite_frac < 1):
        raise ValueError("ga.elite_fraction must be in (0, 1)")
    mut_prob = ga.get("mutation_prob", 0.1)
    if not (0 <= mut_prob <= 1):
        raise ValueError("ga.mutation_prob must be in [0, 1]")

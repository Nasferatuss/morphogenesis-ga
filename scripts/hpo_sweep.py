"""CLI entry point for Optuna-backed HPO sweeps.

Thin wrapper around ``core.services.hpo`` that creates an Optuna study,
suggests parameter sets per trial, runs ``train_ga`` in a temp dir, and
extracts the objective. Results are written to ``runs/hpo/<study_name>/``.

Usage
-----

    python scripts/hpo_sweep.py \\
        --base-config configs/reproducibility_freeze_v1.yaml \\
        --search-space configs/hpo/search_space_v1.yaml \\
        --n-trials 20 \\
        --study-name v1_demo \\
        --objective best_iou

Outputs
-------
- ``runs/hpo/<study>/trials.csv``     — all trials (params + objective)
- ``runs/hpo/<study>/best_params.yaml`` — winner parameters
- ``runs/hpo/<study>/importance.csv`` — param importance (if ≥2 successful trials)
- ``runs/hpo/<study>/study.log``      — stdout of the whole sweep
- ``runs/hpo/<study>/trial_XXXX/``    — per-trial runs/ subdirs
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

# Allow "python scripts/hpo_sweep.py" from repo root to find top-level packages.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import yaml  # noqa: E402

from agents.train_agent.pipeline import train_ga  # noqa: E402
from core.services.hpo import (  # noqa: E402
    VALID_OBJECTIVES,
    load_search_space,
    run_trial,
    suggest_params,
)
from src.targets import make_target  # noqa: E402
from src.utils import select_device, set_seed  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna-backed hyperparameter sweep for morphogenesis-ga.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--search-space", required=True, type=Path)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Defaults to hpo_<timestamp>",
    )
    parser.add_argument(
        "--objective",
        choices=sorted(VALID_OBJECTIVES),
        default="best_iou",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=42,
        help="Optuna sampler seed (separate from GA seed).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optional SQLite URL for resumable studies, e.g. sqlite:///hpo.db",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Wall-clock timeout in seconds (None = unlimited).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/hpo"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import optuna
    except ImportError as err:  # pragma: no cover
        raise SystemExit(
            "[HPO] optuna is required. Install via: pip install -e '.[hpo]'"
        ) from err

    if not args.base_config.exists():
        raise FileNotFoundError(f"Base config not found: {args.base_config}")
    if not args.search_space.exists():
        raise FileNotFoundError(f"Search space not found: {args.search_space}")

    # Load base config + search space
    with args.base_config.open(encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)
    specs = load_search_space(args.search_space)

    # Target setup (mirrors run_train.main)
    seed = int(base_cfg.get("seed", 42))
    set_seed(seed)
    device = select_device(base_cfg.get("device", {}).get("prefer", "cpu"))
    world_cfg = base_cfg.get("world", {})
    target_cfg = base_cfg.get("target", {})
    target_name = str(target_cfg.get("name", "T"))
    width = int(world_cfg.get("width", 15))
    height = int(world_cfg.get("height", 15))
    import torch as _torch
    target_mask = make_target(target_name, height, width)
    target_area = float(target_mask.to(dtype=_torch.bool).sum().item())

    # Study dir
    study_name = args.study_name or f"hpo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    study_dir = args.output_root / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    print(f"[HPO] study_name={study_name}")
    print(f"[HPO] base_config={args.base_config}")
    print(f"[HPO] search_space={args.search_space} ({len(specs)} params)")
    print(f"[HPO] n_trials={args.n_trials} objective={args.objective}")
    print(f"[HPO] output={study_dir}")
    print(f"[HPO] target={target_name} target_area={target_area:.1f}")

    sampler = optuna.samplers.TPESampler(seed=args.sampler_seed)
    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        sampler=sampler,
        direction="maximize",
        load_if_exists=args.storage is not None,
    )

    def objective_fn(trial: "optuna.Trial") -> float:
        params = suggest_params(trial, specs)
        trial_dir = study_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        value = run_trial(
            base_cfg=base_cfg,
            params=params,
            trial_dir=trial_dir,
            objective=args.objective,
            train_ga_fn=train_ga,
            target_mask=target_mask,
            target_area=target_area,
            device=device,
            seed=seed,
            default_target_name=target_name,
        )
        elapsed = time.time() - start
        print(
            f"[HPO] trial {trial.number:4d}  obj={value:.6f}  "
            f"elapsed={elapsed:6.1f}s"
        )
        return value

    start_total = time.time()
    try:
        study.optimize(
            objective_fn,
            n_trials=args.n_trials,
            timeout=args.timeout,
            gc_after_trial=True,
            show_progress_bar=False,
        )
    except KeyboardInterrupt:
        print("[HPO] Interrupted — writing partial results...")

    total_elapsed = time.time() - start_total
    print(f"[HPO] Total elapsed: {total_elapsed:.1f}s")

    # ---- Write artefacts ----
    trials_csv = study_dir / "trials.csv"
    with trials_csv.open("w", newline="", encoding="utf-8") as f:
        if study.trials:
            param_keys = sorted({k for t in study.trials for k in t.params.keys()})
            fieldnames = ["number", "state", "value"] + param_keys
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for t in study.trials:
                row = {
                    "number": t.number,
                    "state": t.state.name,
                    "value": t.value if t.value is not None else "",
                }
                for k in param_keys:
                    row[k] = t.params.get(k, "")
                writer.writerow(row)

    completed = [t for t in study.trials if t.value is not None]
    if completed:
        best = study.best_trial
        print(f"[HPO] Best trial: #{best.number}  value={best.value:.6f}")
        print("[HPO] Best params:")
        for k, v in sorted(best.params.items()):
            print(f"        {k} = {v}")

        best_params_path = study_dir / "best_params.yaml"
        with best_params_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                {
                    "study_name": study_name,
                    "best_trial_number": best.number,
                    "best_value": float(best.value),
                    "objective": args.objective,
                    "params": dict(best.params),
                    "n_trials_completed": len(completed),
                    "n_trials_total": len(study.trials),
                },
                f,
                sort_keys=False,
            )

        # Importance (requires ≥2 completed trials)
        if len(completed) >= 2:
            try:
                importance = optuna.importance.get_param_importances(study)
                imp_csv = study_dir / "importance.csv"
                with imp_csv.open("w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["param", "importance"])
                    for param, imp in importance.items():
                        writer.writerow([param, imp])
                print(f"[HPO] Param importance written to {imp_csv}")
            except Exception as err:
                print(f"[HPO] Importance computation failed: {err}")
    else:
        print("[HPO] No completed trials — nothing to summarise.")

    print(f"[HPO] Artefacts: {study_dir}")


if __name__ == "__main__":
    main()

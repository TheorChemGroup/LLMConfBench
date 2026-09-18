from __future__ import annotations

import sys
from pathlib import Path

from config.models_config import geom_seed_paths, models_by_id, models_in_group
from metrics.grouped_metrics import has_three_complete_seeds


def complete_answer_files(
    deltae_override: dict | None = None,
    seed_dirs: list[Path] | None = None,
) -> set[str]:
    dirs = list(seed_dirs) if seed_dirs is not None else geom_seed_paths()
    if not dirs:
        return set()
    filenames = [
        {path.name for path in seed_dir.glob("answers_*.jsonl")}
        for seed_dir in dirs
        if seed_dir.is_dir()
    ]
    if len(filenames) != len(dirs):
        return set()
    candidates = set.intersection(*filenames)
    return {
        filename
        for filename in candidates
        if has_three_complete_seeds(filename, dirs, deltae_override)
    }


def warn_unregistered_complete_models(
    complete_files: set[str],
) -> None:
    all_specs = models_by_id()
    by_filename = {
        spec.answers: (model_id, spec)
        for model_id, spec in all_specs.items()
        if spec.answers
    }
    group_ids = {spec.id for spec in models_in_group("geom_benchmark")}
    for filename in sorted(complete_files):
        registered = by_filename.get(filename)
        if registered is None:
            suggested_id = filename.removeprefix("answers_").removesuffix(".jsonl")
            print(
                "WARNING: complete 3-seed answers file is not registered in models.yaml: "
                f"{filename}\n"
                f"  Add models.{suggested_id} with display, short, answers: {filename}, "
                "color, class, reasoning, and release_date; then add its id to "
                "groups.geom_benchmark.",
                file=sys.stderr,
            )
            continue
        model_id, spec = registered
        if model_id not in group_ids:
            print(
                "WARNING: complete 3-seed model is registered but omitted from "
                f"groups.geom_benchmark: {model_id} ({spec.display}, {filename})",
                file=sys.stderr,
            )


def complete_geom_models(
    deltae_override: dict | None = None,
    seed_dirs: list[Path] | None = None,
    *,
    warn_unregistered: bool = True,
) -> list[tuple[str, str]]:
    dirs = list(seed_dirs) if seed_dirs is not None else geom_seed_paths()
    complete_files = complete_answer_files(deltae_override, dirs)
    if warn_unregistered:
        warn_unregistered_complete_models(complete_files)
    out: list[tuple[str, str]] = []
    for spec in models_in_group("geom_benchmark"):
        fname = spec.answers
        if not fname:
            continue
        if fname not in complete_files:
            print(f"  skip {spec.display}: not 3 complete seeds", flush=True)
            continue
        out.append((spec.display, fname))
    return out

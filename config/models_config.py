from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from datetime import date, datetime, time
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_YAML = REPO_ROOT / "models.yaml"


ARCH_VALUES: frozenset[str] = frozenset({"dense", "moe", "distill"})
ARCH_MARKERS: dict[str | None, str] = {
    "dense": "o",
    "moe": "^",
    "distill": "D",
    None: "^",
}
MARKER_SIZE_PARAMS_MIN = 18.0
MARKER_SIZE_PARAMS_MAX = 110.0
PARAMS_SIZE_REF_LO_B = 10.0
PARAMS_SIZE_REF_HI_B = 3000.0


@dataclass(frozen=True)
class ModelSpec:
    id: str
    display: str
    short: str
    answers: str | None
    color: str
    model_class: str
    release_date: str | None = None
    reasoning: bool | None = None
    params_total_b: float | None = None
    params_active_b: float | None = None
    arch: str | None = None


def repo_root() -> Path:
    return REPO_ROOT


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    if not MODELS_YAML.is_file():
        raise FileNotFoundError(f"Missing unified model registry: {MODELS_YAML}")
    with MODELS_YAML.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_arch(value: Any) -> str | None:
    if value is None:
        return None
    arch = str(value).strip().lower()
    if arch in {"", "null", "none"}:
        return None
    if arch not in ARCH_VALUES:
        raise ValueError(f"Unknown arch {value!r}; expected one of {sorted(ARCH_VALUES)}")
    return arch


def _model_spec(model_id: str, raw: dict[str, Any]) -> ModelSpec:
    return ModelSpec(
        id=model_id,
        display=str(raw["display"]),
        short=str(raw.get("short", raw["display"])),
        answers=raw.get("answers"),
        color=str(raw.get("color", "#6b7280")),
        model_class=str(raw.get("class", "unknown")),
        release_date=raw.get("release_date"),
        reasoning=raw.get("reasoning"),
        params_total_b=_optional_float(raw.get("params_total_b")),
        params_active_b=_optional_float(raw.get("params_active_b")),
        arch=_optional_arch(raw.get("arch")),
    )


def models_by_id() -> dict[str, ModelSpec]:
    cfg = load_config()
    return {mid: _model_spec(mid, raw) for mid, raw in cfg["models"].items()}


def models_in_group(group: str) -> list[ModelSpec]:
    cfg = load_config()
    by_id = models_by_id()
    ids = cfg["groups"][group]
    out: list[ModelSpec] = []
    for mid in ids:
        if mid not in by_id:
            raise KeyError(f"Group {group!r} references unknown model id {mid!r}")
        out.append(by_id[mid])
    return out


MARKER_DEFAULT = "^"
MARKER_REASONING = "o"
MARKER_SIZE = 80
MARKER_SIZE_LOCAL = 20
MARKER_EDGEWIDTH = 0.9
MARKER_EDGEWIDTH_LOCAL = 0.55
LEGEND_MARKERSIZE = 7
LEGEND_MARKERSIZE_LOCAL = 3.5
CLASS_OS = frozenset({"os", "local"})


TAU_FLOOR = 2.0
TAU_KEY = f"tau_f{TAU_FLOOR}"
CLASS_CLOSED = frozenset({"closed", "frontier"})


PILOT_MOLECULE_IDS: frozenset[str] = frozenset({
    "CC__O_OC1_C_C_C__CCC2OC2_C_CC3OC__O_C__C_C13",
    "C_C__H__O__C_C__C__H__C_OO",
})
PILOT_ROW_LABELS: dict[str, str] = {
    "CC__O_OC1_C_C_C__CCC2OC2_C_CC3OC__O_C__C_C13": "mPM",
    "C_C__H__O__C_C__C__H__C_OO": "mPP",
}


def is_reasoning_or_thinking(display: str) -> bool:
    for m in models_by_id().values():
        if m.display == display and m.reasoning is not None:
            return m.reasoning
    lower = display.lower()
    return "reasoning" in lower or "thinking" in lower


def is_reasoning_model_id(model_id: str) -> bool:
    m = models_by_id()[model_id]
    if m.reasoning is not None:
        return m.reasoning
    return is_reasoning_or_thinking(m.display)


def marker_for_display(display: str) -> str:
    return MARKER_REASONING if is_reasoning_or_thinking(display) else MARKER_DEFAULT


def marker_for_model_id(model_id: str) -> str:
    return MARKER_REASONING if is_reasoning_model_id(model_id) else MARKER_DEFAULT


def tag_display_map(group: str = "geom_benchmark") -> dict[str, str]:
    return {m.id: m.display for m in models_in_group(group)}


def colors_by_id(group: str = "geom_benchmark") -> dict[str, str]:
    return {m.id: m.color for m in models_in_group(group)}


def colors_by_display() -> dict[str, str]:
    return {m.display: m.color for m in models_by_id().values()}


def shorts_by_display() -> dict[str, str]:
    return {m.display: m.short for m in models_by_id().values()}


def color_for_display(display: str, *, default: str = "#6b7280") -> str:
    return colors_by_display().get(display, default)


def class_for_display(display: str, *, default: str = "unknown") -> str:
    for m in models_by_id().values():
        if m.display == display:
            return m.model_class
    return default


def is_os_display(display: str) -> bool:
    return class_for_display(display) in CLASS_OS


def is_closed_display(display: str) -> bool:
    return class_for_display(display) in CLASS_CLOSED


def is_local_display(display: str) -> bool:
    return is_os_display(display)


def scatter_style_for_display(display: str) -> tuple[float, float]:
    if is_local_display(display):
        return MARKER_SIZE_LOCAL, MARKER_EDGEWIDTH_LOCAL
    return MARKER_SIZE, MARKER_EDGEWIDTH


def params_active_for_display(display: str) -> float | None:
    for m in models_by_id().values():
        if m.display == display:
            return m.params_active_b
    return None


def params_total_for_display(display: str) -> float | None:
    for m in models_by_id().values():
        if m.display == display:
            return m.params_total_b
    return None


def arch_for_display(display: str) -> str | None:
    for m in models_by_id().values():
        if m.display == display:
            return m.arch
    return None


def marker_for_arch(arch: str | None) -> str:
    return ARCH_MARKERS.get(arch, ARCH_MARKERS[None])


def scatter_size_for_params(params_b: float | None) -> float:
    if params_b is None or params_b <= 0:
        return float(MARKER_SIZE)
    lo = math.log10(PARAMS_SIZE_REF_LO_B)
    hi = math.log10(PARAMS_SIZE_REF_HI_B)
    t = (math.log10(params_b) - lo) / (hi - lo)
    t = min(1.0, max(0.0, t))
    return MARKER_SIZE_PARAMS_MIN + t * (MARKER_SIZE_PARAMS_MAX - MARKER_SIZE_PARAMS_MIN)


def short_label_for_display(display: str) -> str:
    return shorts_by_display().get(display, display)


VENDOR_FAMILIES: tuple[str, ...] = (
    "Gemini",
    "Gemma",
    "Qwen",
    "GPT",
    "Claude",
    "Grok",
    "Moonshot",
    "DeepSeek",
    "Meta",
)


def family_for_display(display: str) -> str:
    if display.startswith("Gemini"):
        return "Gemini"
    if display.startswith("Gemma"):
        return "Gemma"
    if display.startswith("Qwen"):
        return "Qwen"
    if display.startswith(("GPT", "O4")):
        return "GPT"
    if display.startswith("Claude"):
        return "Claude"
    if display.startswith("Grok"):
        return "Grok"
    if display.startswith("Kimi"):
        return "Moonshot"
    if display.startswith("DeepSeek"):
        return "DeepSeek"
    if display.startswith(("Muse", "Meta", "Llama")):
        return "Meta"
    raise ValueError(f"Unknown vendor family for display name {display!r}")


def family_legend_color(family: str, present_displays: set[str]) -> str:
    for name in sorted(present_displays):
        if family_for_display(name) == family:
            return color_for_display(name)
    return "#6b7280"


def model_files_by_display(group: str = "geom_benchmark") -> dict[str, str]:
    return {
        m.display: m.answers
        for m in models_in_group(group)
        if m.answers is not None
    }


def default_models_table(group: str = "geom_benchmark") -> list[tuple[str, str | None]]:
    random_spec = models_by_id()["random"]
    rows: list[tuple[str, str | None]] = [(random_spec.display, random_spec.answers)]
    rows.extend((m.display, m.answers) for m in models_in_group(group))
    return rows


def release_dates_by_display() -> dict[str, datetime]:
    out: dict[str, datetime] = {}
    for m in models_by_id().values():
        if m.release_date:
            out[m.display] = datetime.combine(date.fromisoformat(m.release_date), time.min)
    return out


def geom_seed_paths() -> list[Path]:
    cfg = load_config()
    seeds = cfg["paths"]["seeds"]
    return [REPO_ROOT / seeds["42"], REPO_ROOT / seeds["43"], REPO_ROOT / seeds["44"]]


def geom_seed_dirs() -> list[tuple[str, Path]]:
    cfg = load_config()
    seeds = cfg["paths"]["seeds"]
    return [
        ("42", REPO_ROOT / seeds["42"]),
        ("43", REPO_ROOT / seeds["43"]),
        ("44", REPO_ROOT / seeds["44"]),
    ]


def path_from_config(key: str) -> Path:
    cfg = load_config()
    return REPO_ROOT / cfg["paths"][key]


def geom_conformers_dir() -> Path:
    return path_from_config("conformers")


def mols_md_path() -> Path:
    return path_from_config("mols_md")


def baseline_ids() -> list[str]:
    return list(load_config()["baselines"].keys())


def resolve_deltae_json(
    baseline: str | None = None,
    deltae_json: Path | None = None,
) -> Path | None:
    if deltae_json is not None:
        return deltae_json.resolve()
    if baseline is None:
        return None
    cfg = load_config()
    bl = cfg["baselines"].get(baseline)
    if bl is None:
        raise ValueError(f"Unknown baseline {baseline!r}; choose from {list(cfg['baselines'])}")
    rel = bl.get("deltae_json")
    if rel is None:
        return None
    return (REPO_ROOT / rel).resolve()


def add_baseline_arguments(ap: argparse.ArgumentParser) -> None:
    cfg = load_config()
    choices = list(cfg["baselines"].keys())
    ap.add_argument(
        "--baseline",
        choices=choices,
        default="gfn2-xtb",
        help="Energy baseline for scoring (default: gfn2-xtb = metadata xTB ΔE)",
    )
    ap.add_argument(
        "--deltae-json",
        type=Path,
        default=None,
        help="Override baseline JSON (mol -> {conf_index: ΔE kcal/mol}); "
        "implies r2scan-style conf-index mapping",
    )

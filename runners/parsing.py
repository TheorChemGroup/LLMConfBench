"""Shared parsing/IO for benchmark runners (Ollama + OpenRouter)."""

import json
import re
from pathlib import Path
from typing import Any


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def extract_json_array(text: str) -> list[Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return obj
    except json.JSONDecodeError:
        pass
    fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if fence:
        try:
            obj = json.loads(fence.group(1))
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, list):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def normalize_label(item: Any, n: int) -> str | None:
    expected = {str(i) for i in range(1, n + 1)}
    if isinstance(item, bool):
        return None
    if isinstance(item, int):
        s = str(item)
        return s if s in expected else None
    if isinstance(item, float) and item == int(item):
        s = str(int(item))
        return s if s in expected else None
    if not isinstance(item, str):
        return None
    s = item.strip()
    if s in expected:
        return s
    m = re.match(r"^(?:[Cc]onformer\s+)?(\d+)\b", s)
    if m:
        t = m.group(1)
        return t if t in expected else None
    return None


def _normalize_ranking_list(ranking: list[Any], n: int) -> list[str] | None:
    expected = {str(i) for i in range(1, n + 1)}
    out: list[str] = []
    for item in ranking:
        lab = normalize_label(item, n)
        if lab is None:
            return None
        out.append(lab)
    if len(out) != n or set(out) != expected:
        return None
    return out


def parse_ranking_n(raw: str, n: int) -> tuple[list[str] | None, dict[str, Any] | None, bool, bool]:
    obj = extract_json_object(raw)
    if obj is not None:
        ranking = obj.get("ranking")
        if not isinstance(ranking, list):
            return None, obj, False, True
        out = _normalize_ranking_list(ranking, n)
        if out is None:
            return None, obj, False, True
        return out, obj, False, False

    arr = extract_json_array(raw)
    if arr is not None:
        out = _normalize_ranking_list(arr, n)
        if out is None:
            return None, None, False, True
        return out, None, False, False

    return None, None, True, False


def load_metadata_jobs(bundle_dir: Path) -> list[dict[str, Any]]:
    meta_path = bundle_dir / "prompt_metadata.jsonl"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing {meta_path}")
    rows: list[dict[str, Any]] = []
    for line in meta_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def resolve_txt_path(bundle_dir: Path, row: dict[str, Any]) -> Path:
    tp = row.get("txt_path")
    if isinstance(tp, str) and tp.strip():
        p = bundle_dir / Path(tp).name
        if p.is_file():
            return p
    mol = str(row["molecule"])
    safe = re.sub(r"[^\w.\-]+", "_", mol).strip("_") or "molecule"
    p = bundle_dir / f"{safe}.txt"
    if p.is_file():
        return p
    raise FileNotFoundError(f"No .txt for molecule {mol!r} under {bundle_dir}")



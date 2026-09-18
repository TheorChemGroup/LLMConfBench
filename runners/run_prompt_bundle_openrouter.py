#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from dotenv import load_dotenv
from tqdm import tqdm

from runners.api_client import (
    REASONING_EFFORTS,
    SUPPORTED_PARAMS,
    ModelPreset,
    call_openrouter,
    get_default_reasoning_effort,
    get_model_reasoning_kind,
    resolve_llm_api_key,
)

SYSTEM_PROMPT = "You are an expert in computational chemistry."
TEMPERATURE = 1.0
SLEEP_BETWEEN = 5.0
TIMEOUT = 1200.0


def _detect_preset(model: str) -> ModelPreset:
    mid = model.strip()
    sp = SUPPORTED_PARAMS.get(mid)
    kind = get_model_reasoning_kind(mid)
    effort = get_default_reasoning_effort(mid)
    return ModelPreset(
        id="custom", model=mid, temperature=TEMPERATURE,
        reasoning_effort=effort, reasoning_kind=kind,
        supported_params=sp,
    )


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


def _extract_usage(api_data: dict | None) -> dict[str, Any]:
    if api_data is None:
        return {}
    usage = api_data.get("usage") or {}
    choices = api_data.get("choices") or []
    finish = choices[0].get("finish_reason") if choices else None
    return {
        "generation_id": api_data.get("id"),
        "cost_total": usage.get("cost"),
        "tokens_prompt": usage.get("prompt_tokens"),
        "tokens_completion": usage.get("completion_tokens"),
        "tokens_reasoning": usage.get("tokens_reasoning"),
        "tokens_cached": usage.get("tokens_cached"),
        "finish_reason_raw": finish,
    }


def run_one(
    row: dict[str, Any],
    user_content: str,
    api_key: str,
    preset: ModelPreset,
    *,
    timeout: float,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    mol = str(row["molecule"])
    n = int(row["n_conformers"])

    llm_text, api_err, api_raw, api_data = call_openrouter(
        api_key, preset, user_content,
        timeout=timeout,
        use_system_prompt=True,
        system_message=SYSTEM_PROMPT,
        max_tokens=max_tokens,
    )
    if api_err:
        row_out = {
            "molecule": mol, "n_conformers": n,
            "prediction": None, "reason": None,
            "parse_error": True, "invalid_ranking": False,
            "api_error": api_err, "raw_response": api_raw,
            "model": preset.model,
        }
        row_out.update(_extract_usage(api_data))
        return row_out

    assert llm_text is not None
    pred, parsed_obj, json_parse_err, invalid_ranking = parse_ranking_n(llm_text, n)
    reason = None
    reasoning_raw = None
    ranking_raw = None
    if isinstance(parsed_obj, dict):
        reason = parsed_obj.get("reasoning") or parsed_obj.get("reason")
        reasoning_raw = parsed_obj.get("reasoning")
        ranking_raw = parsed_obj.get("ranking")

    row_out = {
        "molecule": mol, "n_conformers": n,
        "prediction": pred, "reason": reason,
        "reasoning_raw": reasoning_raw,
        "ranking_raw": ranking_raw,
        "parse_error": json_parse_err, "invalid_ranking": invalid_ranking,
        "api_error": None,
        "raw_response": llm_text,
        "model": preset.model,
    }
    row_out.update(_extract_usage(api_data))
    return row_out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True,
                     help="Directory with prompt_metadata.jsonl and .txt files")
    ap.add_argument("--model", type=str, required=True,
                     help="OpenRouter model ID, e.g. google/gemini-3.6-flash")
    ap.add_argument("--output-tag", type=str, required=True,
                     help="Tag for output files: answers_{tag}.jsonl and predictions_{tag}.jsonl")
    ap.add_argument("--only-molecules", nargs="*", default=None,
                     help="Run only these molecule names (for smoke tests)")
    ap.add_argument("--temperature", type=float, default=None,
                     help="Override temperature (default: 1.0)")
    ap.add_argument("--reasoning", type=str, default=None,
                     choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "enabled"],
                     help="Override reasoning effort (default: auto-detect from model)")
    ap.add_argument("--timeout", type=float, default=None,
                     help="Per-request timeout in seconds (default: 300)")
    ap.add_argument("--sleep", type=float, default=None,
                      help="Seconds between API calls (default: 5)")
    ap.add_argument("--resume-from", type=Path, default=None,
                      help="Skip molecules already in this answers file")
    ap.add_argument("--max-tokens", type=int, default=None,
                      help="Override max_tokens for API requests (default: model default)")
    args = ap.parse_args()

    load_dotenv()
    api_key = resolve_llm_api_key()
    if not api_key:
        print("Set OPENROUTER_API_KEY in .env or environment.", file=sys.stderr)
        return 1

    bundle_dir = args.bundle_dir.resolve()
    preset = _detect_preset(args.model)

    if args.temperature is not None:
        from dataclasses import replace
        preset = replace(preset, temperature=args.temperature)
    if args.reasoning is not None:
        from dataclasses import replace
        if args.reasoning == "none":
            preset = replace(preset, reasoning_kind=None)
        elif args.reasoning == "enabled":
            preset = replace(preset, reasoning_kind="enabled")
        else:
            supported = REASONING_EFFORTS.get(args.model.strip(), [])
            if supported and args.reasoning not in supported:
                print(f"Warning: reasoning effort '{args.reasoning}' may not be supported by {args.model}. "
                      f"Supported: {supported}", file=sys.stderr)
            preset = replace(preset, reasoning_effort=args.reasoning, reasoning_kind="effort")
    timeout = args.timeout or TIMEOUT
    sleep = args.sleep if args.sleep is not None else SLEEP_BETWEEN
    meta_jobs = load_metadata_jobs(bundle_dir)

    jobs = meta_jobs

    if args.resume_from and args.resume_from.is_file():
        existing_mols = set()
        for line in args.resume_from.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                mol = str(obj.get("molecule"))
                rnk = obj.get("ranking")
                if mol and rnk is not None:
                    existing_mols.add(mol)
            except json.JSONDecodeError:
                pass
        jobs_before = len(jobs)
        jobs = [j for j in jobs if str(j["molecule"]) not in existing_mols]
        print(f"Resuming from {args.resume_from}: skipped {len(existing_mols)} done, remaining {len(jobs)}/{jobs_before}")
    if args.only_molecules:
        allow = set(args.only_molecules)
        jobs = [j for j in jobs if str(j["molecule"]) in allow]
        missing = allow - {str(j["molecule"]) for j in jobs}
        if missing:
            print(f"Unknown molecules: {missing}", file=sys.stderr)
            return 1
        if not jobs:
            print("No matching molecules found.", file=sys.stderr)
            return 1

    tag = args.output_tag.strip()
    predictions_path = bundle_dir / f"predictions_{tag}.jsonl"
    answers_path = bundle_dir / f"answers_{tag}.jsonl"

    existing_rows: dict[str, dict[str, Any]] = {}
    if args.resume_from and args.resume_from.is_file():
        for line in args.resume_from.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                mol = str(obj.get("molecule"))
                rnk = obj.get("ranking")
                if mol and rnk is not None:
                    preds_path = bundle_dir / f"predictions_{tag}.jsonl"
                    loaded = False
                    if preds_path.is_file():
                        for pline in preds_path.read_text(encoding="utf-8").splitlines():
                            pline = pline.strip()
                            if not pline:
                                continue
                            try:
                                prow = json.loads(pline)
                                if str(prow.get("molecule")) == mol:
                                    existing_rows[mol] = prow
                                    loaded = True
                                    break
                            except json.JSONDecodeError:
                                pass
                    if not loaded:
                        existing_rows[mol] = obj
            except json.JSONDecodeError:
                pass

    new_rows: dict[str, dict[str, Any]] = {}
    with answers_path.open("a", encoding="utf-8") as af, \
         predictions_path.open("a", encoding="utf-8") as pf:
        for i, row in enumerate(tqdm(jobs, desc=preset.model.split("/")[-1], unit="mol")):
            if i > 0 and sleep > 0:
                time.sleep(sleep)
            mol = str(row["molecule"])
            user_content = resolve_txt_path(bundle_dir, row).read_text(encoding="utf-8")
            new_rows[mol] = run_one(row, user_content, api_key, preset, timeout=timeout, max_tokens=args.max_tokens)
            if new_rows[mol].get("prediction"):
                af.write(json.dumps({"molecule": mol, "ranking": new_rows[mol]["prediction"]}, ensure_ascii=False) + "\n")
            pf.write(json.dumps(new_rows[mol], ensure_ascii=False) + "\n")
            af.flush()
            pf.flush()

    all_rows = {**existing_rows, **new_rows}
    rows_out = [all_rows[str(r["molecule"])] for r in meta_jobs if str(r["molecule"]) in all_rows]

    with predictions_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in rows_out if r.get("prediction") is not None)
    print(f"Wrote {predictions_path.name} and {answers_path.name} ({n_ok}/{len(rows_out)} ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

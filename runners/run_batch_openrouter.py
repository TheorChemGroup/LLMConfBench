#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

from dotenv import load_dotenv

from runners.api_client import (
    SUPPORTED_PARAMS,
    ModelPreset,
    get_default_reasoning_effort,
    get_model_reasoning_kind,
    model_supports,
    resolve_llm_api_key,
)
from runners.run_prompt_bundle_openrouter import (
    SYSTEM_PROMPT,
    load_metadata_jobs,
    parse_ranking_n,
    resolve_txt_path,
)

TIMEOUT = 600.0
POLL_INTERVAL = 30.0
_DEFAULT_BATCH_URL = "https://openrouter.ai/api/v1/beta/batches"


def batch_url() -> str:
    base = os.environ.get("OPENROUTER_BASE_URL", "").strip()
    if not base:
        return _DEFAULT_BATCH_URL
    b = base.rstrip("/")
    return f"{b}/beta/batches"


def _headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/local/benchmark_ranking",
        "X-Title": "Conformer ranking benchmark",
    }


def _detect_preset(model: str) -> ModelPreset:
    sp = SUPPORTED_PARAMS.get(model.strip())
    kind = get_model_reasoning_kind(model)
    effort = get_default_reasoning_effort(model)
    return ModelPreset(
        id="custom", model=model.strip(), temperature=1.0,
        reasoning_effort=effort, reasoning_kind=kind,
        supported_params=sp,
    )


def _body_for_row(preset: ModelPreset, row: dict[str, Any], bundle_dir: Path,
                  max_tokens: int | None) -> dict[str, Any]:
    user_content = resolve_txt_path(bundle_dir, row).read_text(encoding="utf-8")
    body: dict[str, Any] = {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
    }
    if model_supports(preset, "temperature"):
        body["temperature"] = preset.temperature
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    if model_supports(preset, "reasoning"):
        if preset.reasoning_kind == "enabled":
            body["reasoning"] = {"enabled": True}
        else:
            body["reasoning"] = {"effort": preset.reasoning_effort}
    return body


def submit(api_key: str, preset: ModelPreset, jobs: list[dict[str, Any]],
           bundle_dir: Path, max_tokens: int | None) -> dict[str, Any]:
    requests_list = []
    for row in jobs:
        mol = str(row["molecule"])
        requests_list.append({
            "custom_id": mol,
            "body": _body_for_row(preset, row, bundle_dir, max_tokens),
        })
    payload = {
        "endpoint": "/v1/chat/completions",
        "model": preset.model,
        "requests": requests_list,
    }
    r = requests.post(batch_url(), headers=_headers(api_key), json=payload, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fetch_batch(api_key: str, batch_id: str) -> dict[str, Any]:
    r = requests.get(f"{batch_url()}/{batch_id}", headers=_headers(api_key), timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _row_from_result(result: dict[str, Any], n_map: dict[str, int], batch_cost: float | None) -> dict[str, Any]:
    custom_id = str(result.get("custom_id"))
    n = n_map.get(custom_id)
    err = result.get("error")
    resp = result.get("response") or {}
    row_out = {
        "molecule": custom_id, "n_conformers": n,
        "prediction": None, "reason": None,
        "parse_error": False, "invalid_ranking": False,
        "api_error": None, "model": None, "cost_total": batch_cost,
    }
    if err or resp.get("status_code") != 200:
        row_out["api_error"] = json.dumps(err, ensure_ascii=False) if err else f"HTTP {resp.get('status_code')}"
        row_out["raw_response"] = json.dumps(resp, ensure_ascii=False)
        return row_out

    body = resp.get("body") or {}
    row_out["model"] = body.get("model")
    usage = body.get("usage")
    if isinstance(usage, dict):
        row_out.update({
            "tokens_prompt": usage.get("prompt_tokens"),
            "tokens_completion": usage.get("completion_tokens"),
            "tokens_reasoning": usage.get("tokens_reasoning"),
            "tokens_cached": usage.get("tokens_cached"),
        })
    choices = body.get("choices") or []
    if not choices:
        row_out["api_error"] = "No choices in result body"
        row_out["raw_response"] = json.dumps(body, ensure_ascii=False)
        return row_out
    msg = choices[0].get("message") or {}
    content = msg.get("content")
    if content is None:
        row_out["api_error"] = "Empty message content"
        row_out["raw_response"] = json.dumps(body, ensure_ascii=False)
        return row_out
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p)
            for p in content
        )
    content = str(content).strip()
    pred, parsed_obj, json_parse_err, invalid_ranking = parse_ranking_n(content, n)
    row_out["prediction"] = pred
    row_out["parse_error"] = json_parse_err
    row_out["invalid_ranking"] = invalid_ranking
    if isinstance(parsed_obj, dict):
        row_out["reason"] = parsed_obj.get("reasoning") or parsed_obj.get("reason")
    if json_parse_err or invalid_ranking:
        row_out["raw_response"] = content
    return row_out


def process_completed(batch_obj: dict[str, Any], meta_jobs: list[dict[str, Any]],
                      tag: str, bundle_dir: Path) -> None:
    n_map = {str(r["molecule"]): int(r["n_conformers"]) for r in meta_jobs}
    usage = batch_obj.get("usage") or {}
    batch_cost = usage.get("cost")

    answers_path = bundle_dir / f"answers_{tag}.jsonl"
    predictions_path = bundle_dir / f"predictions_{tag}.jsonl"
    rows_out: dict[str, dict[str, Any]] = {}

    for result in batch_obj.get("results") or []:
        custom_id = str(result.get("custom_id"))
        rows_out[custom_id] = _row_from_result(result, n_map, batch_cost)

    with answers_path.open("w", encoding="utf-8") as af, \
         predictions_path.open("w", encoding="utf-8") as pf:
        for row in meta_jobs:
            mol = str(row["molecule"])
            if mol not in rows_out:
                continue
            r = rows_out[mol]
            if r.get("prediction"):
                af.write(json.dumps({"molecule": mol, "ranking": r["prediction"]}, ensure_ascii=False) + "\n")
            pf.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_ok = sum(1 for r in rows_out.values() if r.get("prediction") is not None)
    n_failed = sum(1 for r in rows_out.values() if r.get("api_error"))
    print(f"Wrote {answers_path.name} and {predictions_path.name} "
          f"({n_ok} ok, {n_failed} api errors) of {len(rows_out)}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--model", type=str, default=None,
                    help="OpenRouter model slug; required for submit")
    ap.add_argument("--output-tag", type=str, required=True)
    ap.add_argument("--batch-id", type=str, default=None,
                    help="Batch id (default: read from batch_id_{tag}.txt)")
    ap.add_argument("--reasoning", type=str, default=None,
                    choices=["none", "minimal", "low", "medium", "high", "xhigh", "max", "enabled"],
                    help="Override reasoning effort (default: auto-detect from model)")
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--only-molecules", nargs="*", default=None)
    ap.add_argument("--poll", action="store_true", help="Poll batch once; write results if completed")
    ap.add_argument("--wait", action="store_true", help="Poll in a loop until a terminal status")
    ap.add_argument("--poll-interval", type=float, default=POLL_INTERVAL)
    args = ap.parse_args()

    load_dotenv()
    api_key = resolve_llm_api_key()
    if not api_key:
        print("Set OPENROUTER_API_KEY in .env or environment.", file=sys.stderr)
        return 1

    bundle_dir = args.bundle_dir.resolve()
    tag = args.output_tag.strip()
    batch_id_file = bundle_dir / f"batch_id_{tag}.txt"

    if not args.poll and not args.wait:
        if not args.model:
            print("--model is required to submit a batch.", file=sys.stderr)
            return 1
        preset = _detect_preset(args.model)
        if args.reasoning is not None:
            from dataclasses import replace
            if args.reasoning == "none":
                preset = replace(preset, reasoning_kind=None)
            elif args.reasoning == "enabled":
                preset = replace(preset, reasoning_kind="enabled")
            else:
                preset = replace(preset, reasoning_effort=args.reasoning, reasoning_kind="effort")
        meta_jobs = load_metadata_jobs(bundle_dir)
        jobs = meta_jobs
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
        print(f"Submitting {len(jobs)} requests as batch model={preset.model} ...")
        resp = submit(api_key, preset, jobs, bundle_dir, args.max_tokens)
        bid = resp.get("id")
        if not bid:
            print(f"Unexpected response: {json.dumps(resp, ensure_ascii=False)}", file=sys.stderr)
            return 1
        batch_id_file.write_text(bid, encoding="utf-8")
        print(f"Batch submitted: id={bid} status={resp.get('status')}")
        print(f"Saved batch id to {batch_id_file}")
        print(f"Poll with: python runners/run_batch_openrouter.py --wait "
              f"--bundle-dir {bundle_dir} --output-tag {tag}")
        return 0

    if not args.batch_id:
        if not batch_id_file.is_file():
            print(f"No batch id: pass --batch-id or run submit first ({batch_id_file}).", file=sys.stderr)
            return 1
        args.batch_id = batch_id_file.read_text(encoding="utf-8").strip()

    meta_jobs = load_metadata_jobs(bundle_dir)
    terminal = {"completed", "failed", "expired", "cancelled"}

    while True:
        batch_obj = fetch_batch(api_key, args.batch_id)
        status = batch_obj.get("status")
        counts = batch_obj.get("request_counts") or {}
        print(f"status={status} counts={json.dumps(counts)}")
        if status == "completed":
            process_completed(batch_obj, meta_jobs, tag, bundle_dir)
            return 0
        if status in terminal:
            print(f"Batch reached terminal status: {status}", file=sys.stderr)
            return 1
        if not args.wait:
            return 0
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    raise SystemExit(main())

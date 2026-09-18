#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import requests
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

SYSTEM_PROMPT = "You are an expert in computational chemistry."
TEMPERATURE = 1.0
SLEEP_BETWEEN = 1.0
MAX_RETRIES = 10
TIMEOUT = 7200.0
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_URL = f"{OLLAMA_HOST}/v1/chat/completions"
OLLAMA_SHOW_URL = f"{OLLAMA_HOST}/api/show"


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


def parse_ranking_n(raw: str, n: int) -> tuple[list[str] | None, dict[str, Any] | None, bool, bool]:
    obj = extract_json_object(raw)
    if obj is None:
        return None, None, True, False
    ranking = obj.get("ranking")
    if not isinstance(ranking, list):
        return None, obj, False, True
    expected = {str(i) for i in range(1, n + 1)}
    out: list[str] = []
    for item in ranking:
        lab = normalize_label(item, n)
        if lab is None:
            return None, obj, False, True
        out.append(lab)
    if len(out) != n or set(out) != expected:
        return None, obj, False, True
    return out, obj, False, False


def probe_thinking(model: str) -> tuple[bool, bool]:
    try:
        resp = requests.post(OLLAMA_SHOW_URL, json={"model": model}, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        print(f"ollama /api/show failed ({e}); skipping think/reasoning_effort",
              file=sys.stderr)
        return False, False
    caps = {str(c).lower() for c in (data.get("capabilities") or [])}
    has_think = "thinking" in caps
    return has_think, has_think


def call_ollama(
    model: str,
    user_content: str,
    *,
    system_message: str = SYSTEM_PROMPT,
    temperature: float = TEMPERATURE,
    timeout: float = TIMEOUT,
    think: bool = False,
    reasoning_effort: bool = False,
) -> tuple[str | None, str | None, dict[str, Any] | None, str | None]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ],
        "stream": False,
        "options": {"temperature": temperature},
    }
    if think:
        payload["think"] = True
    if reasoning_effort:
        payload["reasoning_effort"] = "high"
    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        msg = data.get("choices", [{}])[0].get("message", {})
        text = msg.get("content", "")
        reasoning = msg.get("reasoning_content") or msg.get("reasoning")
        return text, None, data, reasoning
    except requests.exceptions.Timeout:
        return None, "timeout", None, None
    except requests.exceptions.ConnectionError:
        return None, "connection_error — is Ollama running?", None, None
    except requests.RequestException as e:
        return None, str(e), None, None


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


def run_one(
    row: dict[str, Any],
    user_content: str,
    model: str,
    *,
    temperature: float,
    timeout: float,
    think: bool,
    reasoning_effort: bool,
) -> dict[str, Any]:
    mol = str(row["molecule"])
    n = int(row["n_conformers"])

    llm_text, api_err, _, llm_reasoning = call_ollama(
        model, user_content,
        temperature=temperature,
        timeout=timeout,
        think=think,
        reasoning_effort=reasoning_effort,
    )
    if api_err:
        return {
            "molecule": mol, "n_conformers": n,
            "prediction": None, "reason": None,
            "parse_error": True, "invalid_ranking": False,
            "api_error": api_err, "raw_response": None,
            "model": model,
        }

    assert llm_text is not None
    pred, parsed_obj, json_parse_err, invalid_ranking = parse_ranking_n(llm_text, n)
    reason = None
    reasoning_raw = None
    ranking_raw = None
    if isinstance(parsed_obj, dict):
        reason = parsed_obj.get("reasoning") or parsed_obj.get("reason")
        reasoning_raw = parsed_obj.get("reasoning")
        ranking_raw = parsed_obj.get("ranking")

    return {
        "molecule": mol, "n_conformers": n,
        "prediction": pred, "reason": reason,
        "reasoning_raw": reasoning_raw,
        "ranking_raw": ranking_raw,
        "reasoning_content": llm_reasoning,
        "raw_response": llm_text,
        "parse_error": json_parse_err, "invalid_ranking": invalid_ranking,
        "api_error": None,
        "model": model,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bundle-dir", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True,
                     help="Ollama model name, e.g. deepseek-r1:32b")
    ap.add_argument("--output-tag", type=str, required=True)
    ap.add_argument("--only-molecules", nargs="*", default=None)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--timeout", type=float, default=TIMEOUT)
    ap.add_argument("--sleep", type=float, default=SLEEP_BETWEEN)
    ap.add_argument("--resume-from", type=Path, default=None,
                     help="Skip molecules already in this answers file")
    args = ap.parse_args()

    bundle_dir = args.bundle_dir.resolve()

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
        print(f"Resuming: skipped {len(existing_mols)} done, remaining {len(jobs)}/{jobs_before}")

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

    use_think, use_effort = probe_thinking(args.model)
    print(
        f"{args.model}: think={'true' if use_think else 'skip'} "
        f"reasoning_effort={'high' if use_effort else 'skip'}"
    )

    n_ok = 0
    n_total = 0
    with answers_path.open("a", encoding="utf-8") as af, \
         predictions_path.open("a", encoding="utf-8") as pf:
        for i, row in enumerate(tqdm(jobs, desc=args.model, unit="mol")):
            if i > 0 and args.sleep > 0:
                time.sleep(args.sleep)
            mol = str(row["molecule"])
            user_content = resolve_txt_path(bundle_dir, row).read_text(encoding="utf-8")

            result = None
            for attempt in range(1, MAX_RETRIES + 1):
                result = run_one(
                    row, user_content, args.model,
                    temperature=args.temperature, timeout=args.timeout,
                    think=use_think, reasoning_effort=use_effort,
                )
                if result["prediction"] is not None:
                    n_ok += 1
                    break
                if attempt < MAX_RETRIES:
                    if result["api_error"]:
                        print(
                            f"\n{mol}: API error: {result['api_error']}, "
                            f"retrying ({attempt}/{MAX_RETRIES})..."
                        )
                    else:
                        print(f"\n{mol}: INVALID ranking, retrying ({attempt}/{MAX_RETRIES})...")
                    time.sleep(0.1)
                elif result["api_error"]:
                    print(f"\n{mol}: API error after {MAX_RETRIES} attempts: {result['api_error']}")
                else:
                    print(f"\n{mol}: INVALID ranking after {MAX_RETRIES} attempts")
            n_total += 1

            af.write(json.dumps({"molecule": mol, "ranking": result["prediction"]}, ensure_ascii=False) + "\n")
            pf.write(json.dumps(result, ensure_ascii=False) + "\n")
            af.flush()
            pf.flush()

    print(f"Wrote {predictions_path.name} and {answers_path.name} ({n_ok}/{n_total} ok)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Energy leak probe via OpenRouter: same prompt/jobs as energy_probe.py, sent to OpenRouter."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import runners.energy_probe as ep
from config.models_config import geom_conformers_dir
from runners.api_client import (
    SUPPORTED_PARAMS,
    ModelPreset,
    call_openrouter,
    get_default_reasoning_effort,
    get_model_reasoning_kind,
    resolve_llm_api_key,
)

SYSTEM_PROMPT = "You are an expert in computational chemistry."


def detect_preset(model: str, temperature: float = 1.0) -> ModelPreset:
    sp = SUPPORTED_PARAMS.get(model.strip())
    kind = get_model_reasoning_kind(model)
    effort = get_default_reasoning_effort(model)
    return ModelPreset(
        id="custom",
        model=model.strip(),
        temperature=temperature,
        reasoning_effort=effort,
        reasoning_kind=kind,
        supported_params=sp,
    )


def extract_usage(api_data: dict | None) -> dict[str, Any]:
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


def extract_reasoning(api_data: dict | None) -> str | None:
    if not api_data:
        return None
    choices = api_data.get("choices") or []
    msg = (choices[0] or {}).get("message") or {}
    reasoning = msg.get("reasoning")
    if reasoning is None:
        reasoning = msg.get("reasoning_content")
    return reasoning


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="anthropic/claude-sonnet-4.5",
                    help="OpenRouter model id (e.g. openai/gpt-5, google/gemini-2.5-flash)")
    ap.add_argument(
        "--confs-dir",
        type=Path,
        default=None,
        help="Default: geom_conformers_dir() from models.yaml",
    )
    ap.add_argument(
        "--bench-json",
        type=Path,
        default=ep._REPO / "geom-qm9/molecules.json",
        help="JSON with molecules[].id list",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL (default: geom-qm9/energy_probe_or_<tag>.jsonl)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--sleep", type=float, default=5.0)
    ap.add_argument("--max-tokens", type=int, default=None)
    ap.add_argument("--no-reasoning", action="store_true",
                    help="Omit the reasoning/request-flags (no 'reasoning' field in API body)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Debug: first N molecules only")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    load_dotenv()
    api_key = resolve_llm_api_key()
    if not api_key:
        print("Set OPENROUTER_API_KEY in .env or environment.", file=sys.stderr)
        return 1

    confs_dir = args.confs_dir or geom_conformers_dir()
    if not confs_dir.is_dir():
        raise FileNotFoundError(f"conformers dir not found: {confs_dir}")

    preset = detect_preset(args.model, temperature=args.temperature)
    out_path = args.out or (ep._REPO / "geom-qm9" / f"energy_probe_or_{ep.model_tag(args.model)}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    molecule_ids = ep.load_molecule_ids(args.bench_json)
    if args.limit is not None:
        molecule_ids = molecule_ids[: args.limit]

    jobs = ep.build_jobs(molecule_ids, confs_dir, args.seed)
    done = {} if args.no_resume else ep.load_done(out_path)

    print(f"{preset.model}: reasoning_kind={preset.reasoning_kind} "
          f"reasoning_effort={preset.reasoning_effort} (omitted={args.no_reasoning})")
    print(f"conformers dir: {confs_dir}")
    print(f"random conf seed: {args.seed}")
    print(f"jobs: {len(jobs)}  already done: {len(done)}")
    print(f"out: {out_path}")

    mode = "w" if args.no_resume else "a"
    with out_path.open(mode, encoding="utf-8") as out_f:
        for mol_id, conf_idx, xyz_file in tqdm(jobs, desc="energy probe (openrouter)"):
            if mol_id in done:
                continue

            truth_e = ep.parse_truth_energy_kcal(xyz_file)
            xyz_block = ep.load_xyz_no_energy(xyz_file)
            user_content = f"{ep.DEFAULT_PROMPT}\n\n{xyz_block}"

            raw, api_err, _, api_data = None, "not_started", None, None
            for attempt in range(1, args.max_retries + 1):
                raw, api_err, _, api_data = call_openrouter(
                    api_key,
                    preset,
                    user_content,
                    timeout=args.timeout,
                    use_system_prompt=True,
                    system_message=SYSTEM_PROMPT,
                    omit_reasoning=args.no_reasoning,
                    max_tokens=args.max_tokens,
                )
                if not api_err:
                    break
                print(f"{mol_id}: {api_err} (attempt {attempt}/{args.max_retries})")
                time.sleep(args.sleep)

            pred_e = ep.parse_predicted_energy(raw)
            rec = {
                "molecule": mol_id,
                "conf_idx": conf_idx,
                "model": preset.model,
                "xyz_file": str(xyz_file.relative_to(ep._REPO)),
                "truth_energy_kcal": truth_e,
                "predicted_energy_kcal": pred_e,
                "abs_error_kcal": abs(pred_e - truth_e) if pred_e is not None else None,
                "api_error": api_err,
                "reasoning_content": extract_reasoning(api_data),
                "raw_response": raw,
            }
            rec.update(extract_usage(api_data))
            if api_err:
                continue

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            done[mol_id] = rec
            time.sleep(args.sleep)

    rows = [done[mol_id] for mol_id, _, _ in jobs if mol_id in done]
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"\n{preset.model} energy probe (openrouter) · {n}/{len(jobs)} · {out_path.name}")
    if n:
        ok = df["predicted_energy_kcal"].notna()
        print(f"parsed numbers: {int(ok.sum())}/{n}")
        if ok.sum() >= 2:
            err = df.loc[ok, "abs_error_kcal"]
            print(f"MAE: {err.mean():.2f} kcal/mol  median: {err.median():.2f}")
            r = np.corrcoef(
                df.loc[ok, "truth_energy_kcal"],
                df.loc[ok, "predicted_energy_kcal"],
            )[0, 1]
            print(f"Pearson r (truth vs predicted): {r:.3f}")
        print(
            df[
                [
                    "molecule",
                    "conf_idx",
                    "truth_energy_kcal",
                    "predicted_energy_kcal",
                    "abs_error_kcal",
                ]
            ]
            .sort_values(["molecule", "conf_idx"])
            .to_string(index=False)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

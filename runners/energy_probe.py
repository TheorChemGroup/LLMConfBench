"""Energy leak probe: model estimates conformational ΔE from XYZ (no Energy: line)."""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import runners.run_ollama as ro
from config.models_config import geom_conformers_dir

ENERGY_RE = re.compile(r"Energy:\s*([-\d.eE+]+)")

DEFAULT_PROMPT = (
    "Below are Cartesian coordinates of ONE gas-phase conformer (no SMILES, name, or energy).\n\n"
    "Estimate how many kcal/mol HIGHER this conformer is than the global minimum "
    "of the same molecule.\n\n"
    "Use whatever you can:\n"
    "1. Infer the molecular formula from the atom list.\n"
    "2. Reconstruct connectivity / bonding from the geometry.\n"
    "3. If you recognize the compound or recall similar molecules from your training "
    "data (GEOM, QM9, literature, etc.), use that knowledge.\n"
    "4. If you cannot identify it, estimate from sterics and conformational reasoning.\n\n"
    "Reply with one non-negative number in kcal/mol. Nothing else."
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="muse-glimmer:30b")
    ap.add_argument(
        "--confs-dir",
        type=Path,
        default=None,
        help="Default: geom_conformers_dir() from models.yaml",
    )
    ap.add_argument(
        "--bench-json",
        type=Path,
        default=_REPO / "geom-qm9/molecules.json",
        help="JSON with molecules[].id list",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSONL (default: geom-qm9/energy_probe_<tag>.jsonl)",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-retries", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--timeout", type=float, default=3600.0)
    ap.add_argument("--sleep", type=float, default=0.5)
    ap.add_argument("--ollama-host", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="Debug: first N molecules only")
    return ap.parse_args()


def ollama_reachable(host: str) -> bool:
    try:
        requests.get(f"{host}/api/tags", timeout=5).raise_for_status()
        return True
    except requests.RequestException:
        return False


def parse_truth_energy_kcal(path: Path) -> float:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 2:
        raise ValueError(f"bad xyz: {path}")
    m = ENERGY_RE.search(lines[1])
    if not m:
        raise ValueError(f"no Energy: line in {path}")
    return float(m.group(1))


def load_xyz_no_energy(path: Path) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) >= 2 and lines[1].strip().lower().startswith("energy"):
        lines = [lines[0], *lines[2:]]
    return "\n".join(lines).strip()


def list_conf_paths(confs_dir: Path, mol_id: str) -> list[tuple[int, Path]]:
    mol_dir = confs_dir / mol_id
    pat = re.compile(rf"^{re.escape(mol_id)}_conf_(\d+)\.xyz$")
    out: list[tuple[int, Path]] = []
    for p in sorted(mol_dir.glob(f"{mol_id}_conf_*.xyz")):
        m = pat.match(p.name)
        if m:
            out.append((int(m.group(1)), p))
    return out


def parse_predicted_energy(text: str | None) -> float | None:
    if not text:
        return None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text.replace(",", ""))
    if not nums:
        return None
    return float(nums[0])


def model_tag(model: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "", model)


def load_molecule_ids(bench_json: Path) -> list[str]:
    payload = json.loads(bench_json.read_text(encoding="utf-8"))
    return [m["id"] for m in payload["molecules"]]


def build_jobs(
    molecule_ids: list[str],
    confs_dir: Path,
    seed: int,
) -> list[tuple[str, int, Path]]:
    rng = random.Random(seed)
    jobs: list[tuple[str, int, Path]] = []
    for mol_id in molecule_ids:
        choices = list_conf_paths(confs_dir, mol_id)
        if not choices:
            raise FileNotFoundError(f"no conformers in {confs_dir / mol_id}")
        conf_idx, xyz_file = rng.choice(choices)
        jobs.append((mol_id, conf_idx, xyz_file))
    return jobs


def load_done(out_path: Path) -> dict[str, dict]:
    done: dict[str, dict] = {}
    if not out_path.is_file():
        return done
    for line in out_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("api_error"):
            continue
        done[str(rec["molecule"])] = rec
    return done


def main() -> int:
    args = parse_args()
    confs_dir = args.confs_dir or geom_conformers_dir()
    if not confs_dir.is_dir():
        raise FileNotFoundError(f"conformers dir not found: {confs_dir}")

    host = args.ollama_host or ro.OLLAMA_HOST
    ro.OLLAMA_HOST = host
    ro.OLLAMA_URL = f"{host}/v1/chat/completions"
    ro.OLLAMA_SHOW_URL = f"{host}/api/show"

    out_path = args.out or (_REPO / "geom-qm9" / f"energy_probe_{model_tag(args.model)}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    molecule_ids = load_molecule_ids(args.bench_json)
    if args.limit is not None:
        molecule_ids = molecule_ids[: args.limit]

    jobs = build_jobs(molecule_ids, confs_dir, args.seed)
    done = {} if args.no_resume else load_done(out_path)

    if not ollama_reachable(host):
        raise RuntimeError(f"Ollama not reachable at {host}")

    use_think, use_effort = ro.probe_thinking(args.model)
    print(f"{args.model}: think={'on' if use_think else 'off'} reasoning_effort={'high' if use_effort else 'off'}")
    print(f"conformers dir: {confs_dir}")
    print(f"random conf seed: {args.seed}")
    print(f"jobs: {len(jobs)}  already done: {len(done)}")
    print(f"out: {out_path}")

    mode = "w" if args.no_resume else "a"
    with out_path.open(mode, encoding="utf-8") as out_f:
        for mol_id, conf_idx, xyz_file in tqdm(jobs, desc="energy probe"):
            if mol_id in done:
                continue

            truth_e = parse_truth_energy_kcal(xyz_file)
            xyz_block = load_xyz_no_energy(xyz_file)
            user_content = f"{DEFAULT_PROMPT}\n\n{xyz_block}"

            raw, api_err, _, reasoning_content = None, "not_started", None, None
            for attempt in range(1, args.max_retries + 1):
                raw, api_err, _, reasoning_content = ro.call_ollama(
                    args.model,
                    user_content,
                    system_message="You are an expert in computational chemistry.",
                    temperature=args.temperature,
                    timeout=args.timeout,
                    think=use_think,
                    reasoning_effort=use_effort,
                )
                if not api_err:
                    break
                print(f"{mol_id}: {api_err} (attempt {attempt}/{args.max_retries})")
                time.sleep(args.sleep)

            pred_e = parse_predicted_energy(raw)
            rec = {
                "molecule": mol_id,
                "conf_idx": conf_idx,
                "model": args.model,
                "xyz_file": str(xyz_file.relative_to(_REPO)),
                "truth_energy_kcal": truth_e,
                "predicted_energy_kcal": pred_e,
                "abs_error_kcal": abs(pred_e - truth_e) if pred_e is not None else None,
                "api_error": api_err,
                "reasoning_content": reasoning_content,
                "raw_response": raw,
            }
            if api_err:
                continue

            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            done[mol_id] = rec
            time.sleep(args.sleep)

    rows = [done[mol_id] for mol_id, _, _ in jobs if mol_id in done]
    df = pd.DataFrame(rows)
    n = len(df)
    print(f"\n{args.model} energy probe · {n}/{len(jobs)} · {out_path.name}")
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

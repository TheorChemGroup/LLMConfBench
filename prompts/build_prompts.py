#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from config.models_config import geom_conformers_dir, geom_seed_paths


def conformer_labels(n: int) -> list[str]:
    if n < 1:
        raise ValueError("n_confs must be >= 1")
    return [str(i) for i in range(1, n + 1)]


def map_public_labels_to_confs(
    delta_by_idx: dict[str, float],
    xyz_by_idx: dict[str, str],
    labels: list[str],
    rng: random.Random,
) -> tuple[dict[str, float], dict[str, str], dict[str, int]]:
    n = len(labels)
    perm = list(range(n))
    rng.shuffle(perm)
    letter_to_delta: dict[str, float] = {}
    xyz_by_public: dict[str, str] = {}
    label_to_conf: dict[str, int] = {}
    for i in range(n):
        lab = labels[i]
        phys = perm[i]
        s = str(phys)
        letter_to_delta[lab] = delta_by_idx[s]
        xyz_by_public[lab] = xyz_by_idx[s]
        label_to_conf[lab] = phys
    return letter_to_delta, xyz_by_public, label_to_conf


def build_prompt_template(n: int, labels: list[str]) -> str:
    assert len(labels) == n
    label_csv = ", ".join(labels)
    first, last = labels[0], labels[-1]
    return f"""You are given {n} conformers of the same molecule in XYZ format (same connectivity, different coordinates).

Each structure has a fixed numeric label: {label_csv} in the headers [Conformer {first}] ... [Conformer {last}]. Labels are arbitrary identifiers—do not assume that smaller or larger numbers, or numeric order, says anything about stability. The order of blocks below is random and MUST NOT be used as a hint for relative stability.

There are no thermodynamic quantities in the files. Your task is to rank the {n} structures by estimated relative stability (most stable first, least stable last) using reasonable chemical and steric arguments based on the coordinates (e.g. ring strain, eclipsing interactions, close nonbonded contacts, hydrogen-bond donor/acceptor proximity, dipole alignment). If differences are subtle, still output a full ranking and say what is uncertain.

Task: Rank from most stable to least stable. Keep the explanation tied to geometry.

Output: a single JSON object only—no markdown fences, no text before or after.

Because generation is sequential, put \"reasoning\" FIRST (step-by-step analysis of geometry, strain, contacts, uncertainty), then \"ranking\" LAST so the permutation follows your analysis rather than rationalizing a ranking already written.

Schema (use exactly {n} entries in "ranking", each label from {first} to {last} exactly once). Use JSON **strings** for labels, e.g. \"3\" not bare numbers:
{{
  "reasoning": "Analyze hydrogen bonds, steric clashes, ring strain, etc. step-by-step BEFORE committing to the final order.",
  "ranking": ["{first}", "{last}", "..."]
}}

Rules: "ranking" must be a permutation of {{{label_csv}}} (only these numeric string tokens, not the words \"Conformer ...\").

__BLOCKS__"""


def blank_xyz_second_line(xyz: str) -> str:
    lines = xyz.replace("\r\n", "\n").split("\n")
    if len(lines) >= 2:
        lines[1] = ""
    return "\n".join(lines)


def normalize_xyz_geom_style(xyz: str) -> str:
    """Rewrite atom lines to geom-qm9 XYZ style: ``C    0.12345678  -0.12345678   0.12345678``.

    Clears CREST/xTB leading-space / wide-column formatting so pilot prompts match
    the GEOM cohort byte-style. Does not change atom order or coordinates.
    """
    lines = xyz.replace("\r\n", "\n").split("\n")
    if len(lines) < 3:
        return xyz
    try:
        n_atoms = int(lines[0].strip())
    except ValueError as exc:
        raise ValueError(f"Invalid XYZ atom count: {lines[0]!r}") from exc
    out = [lines[0].strip(), lines[1]]
    atom_lines = lines[2 : 2 + n_atoms]
    if len(atom_lines) != n_atoms:
        raise ValueError(f"Expected {n_atoms} atom lines, found {len(atom_lines)}")
    for line in atom_lines:
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"Bad atom line: {line!r}")
        symbol, x, y, z = parts[0], float(parts[1]), float(parts[2]), float(parts[3])


        out.append(f"{symbol:<2} {x:12.8f} {y:12.8f} {z:12.8f}")

    out.extend(lines[2 + n_atoms :])
    return "\n".join(out)


def parse_energy_kcal(line2: str) -> float:
    m = re.search(r"Energy:\s*([-\d.eE+]+)", line2)
    if not m:
        raise ValueError(f"Cannot parse energy from XYZ comment line: {line2!r}")
    return float(m.group(1))


def load_conformers(mol_dir: Path, n: int) -> tuple[dict[str, float], dict[str, str]]:
    pat = re.compile(r"_conf_(\d+)\.xyz$")
    by_idx: dict[int, tuple[float, str]] = {}
    for p in mol_dir.glob("*.xyz"):
        m = pat.search(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        raw = p.read_text(encoding="utf-8", errors="replace")
        lines = raw.replace("\r\n", "\n").split("\n")
        if len(lines) < 2:
            raise ValueError(f"Invalid XYZ: {p}")
        de = parse_energy_kcal(lines[1])
        xyz = normalize_xyz_geom_style(blank_xyz_second_line(raw))
        by_idx[idx] = (de, xyz)
    if len(by_idx) != n:
        raise ValueError(f"Expected {n} *_conf_*.xyz in {mol_dir}, found {len(by_idx)}")
    for k in range(n):
        if k not in by_idx:
            raise ValueError(f"Missing conf index {k} in {mol_dir}")
    delta_by_label: dict[str, float] = {}
    xyz_by_label: dict[str, str] = {}
    for k in range(n):
        de, xyz = by_idx[k]
        ks = str(k)
        delta_by_label[ks] = de
        xyz_by_label[ks] = xyz
    return delta_by_label, xyz_by_label


def build_prompt_blocks(shuffled: list[tuple[str, str]]) -> str:
    parts: list[str] = []
    for label, xyz in shuffled:
        parts.append(f"[Conformer {label}]\n{xyz.strip()}")
    return "\n\n".join(parts)


def shuffle_conformers(
    xyz_by_public_label: dict[str, str], rng: random.Random, labels: list[str]
) -> tuple[list[tuple[str, str]], list[str]]:
    pairs = [(lab, xyz_by_public_label[lab]) for lab in labels]
    rng.shuffle(pairs)
    order_submitted = [p[0] for p in pairs]
    return pairs, order_submitted


def molecule_dirs(data_dir: Path) -> list[Path]:
    return sorted(
        p for p in data_dir.iterdir()
        if p.is_dir() and not p.name.startswith(".")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Build N-conformer prompts (no energies in text).")
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=geom_conformers_dir(),
        help="Directory with one subfolder per molecule (each with *_conf_0 ... *_conf_{N-1}). "
             "Default: geom-qm9/output_30_confs from models.yaml.",
    )
    ap.add_argument(
        "--n-confs",
        type=int,
        default=30,
        metavar="N",
        help="Number of conformers per molecule. Requires *_conf_0 ... *_conf_{N-1}. Labels in prompts are \"1\"...\"N\".",
    )
    ap.add_argument(
        "--molecules",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Molecule folder names under data-dir. "
             "Default: every subfolder of data-dir (all 30 mols).",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=geom_seed_paths()[0],
        help="Output folder (default: seed-42 bundle from models.yaml).",
    )
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for shuffling")
    args = ap.parse_args()

    n_confs = args.n_confs
    labels = conformer_labels(n_confs)
    prompt_template = build_prompt_template(n_confs, labels)

    data_dir: Path = args.data_dir
    if not data_dir.is_dir():
        raise SystemExit(f"Not a directory: {data_dir}")

    if args.molecules is None:
        chosen = molecule_dirs(data_dir)
        if not chosen:
            raise SystemExit(f"No molecule folders in {data_dir}")
    else:
        chosen = []
        for name in args.molecules:
            p = data_dir / name
            if not p.is_dir():
                raise SystemExit(f"Missing molecule directory: {p}")
            chosen.append(p)
    print(f"{len(chosen)} molecule(s) from {data_dir}", flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    meta_path = args.out_dir / "prompt_metadata.jsonl"

    with meta_path.open("w", encoding="utf-8") as meta_f:
        for mol_dir in chosen:
            mol_name = mol_dir.name
            delta_by_idx, xyz_by_idx = load_conformers(mol_dir, n_confs)
            letter_to_delta, xyz_by_public, label_to_conf = map_public_labels_to_confs(
                delta_by_idx, xyz_by_idx, labels, rng,
            )
            safe = re.sub(r"[^\w.\-]+", "_", mol_name).strip("_") or "molecule"
            shuffled, order_submitted = shuffle_conformers(xyz_by_public, rng, labels)
            blocks = build_prompt_blocks(shuffled)
            user_prompt = prompt_template.replace("__BLOCKS__", blocks)

            txt_path = args.out_dir / f"{safe}.txt"
            txt_path.write_text(user_prompt, encoding="utf-8")

            row = {
                "molecule": mol_name,
                "n_conformers": n_confs,
                "txt_path": str(txt_path),
                "seed": args.seed,
                "shuffle_public_labels": True,
                "public_label_to_conf_index": label_to_conf,
                "block_order_in_prompt": order_submitted,
                "letter_to_deltaE_kcal_mol": letter_to_delta,
            }
            print(f"Wrote {txt_path}")
            meta_f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

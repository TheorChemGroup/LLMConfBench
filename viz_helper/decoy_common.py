#!/usr/bin/env python3

"""Shared machinery for H-bond / gauche pair-classification analyses.

Single source for: geometry reading, geometric H-bond detection/counting,
pair-table construction (H-bond buckets, gauche-vs-anti traps, contrast
triples) and all-model pair scoring over the three seed bundles.

Used by plot_equal_hbond_vs_release.py and plot_eqhb_vs_benchmarks.py.
"""

from __future__ import annotations

import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from config.models_config import BUNDLE_SEEDS, geom_conformers_dir, resolve_deltae_json
from metrics.kendall_tau_ranking import (
    augment_meta,
    load_deltae_override,
    load_metadata_index,
)

BUNDLES = list(BUNDLE_SEEDS)
FLOOR = 2.0
COV = {"H": 0.31, "C": 0.76, "N": 0.71, "O": 0.66, "F": 0.57}


def read_xyz(path: Path) -> list[tuple]:
    atoms = []
    for ln in Path(path).read_text().splitlines()[2:]:
        p = ln.split()
        if len(p) >= 4:
            atoms.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return atoms


def dist(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a[1:], b[1:])))


def bonded_graph(atoms) -> dict[int, set[int]]:
    bonded: dict[int, set[int]] = defaultdict(set)
    for i, a in enumerate(atoms):
        for j, b in enumerate(atoms):
            if j <= i:
                continue
            if dist(a, b) < COV.get(a[0], 1.0) + COV.get(b[0], 1.0) + 0.4:
                bonded[i].add(j)
                bonded[j].add(i)
    return bonded


def hbond_contacts(atoms, bonded, *, h_max: float = 2.5,
                   ang_min: float = 120.0) -> list[tuple[float, float]]:
    """Donor-H...O/N contacts as (H...A distance, D-H...A angle)."""
    xyz = {i: a[1:] for i, a in enumerate(atoms)}
    out = []
    for i, a in enumerate(atoms):
        if a[0] != "H":
            continue
        donors = [j for j in bonded.get(i, []) if atoms[j][0] in ("O", "N")]
        if len(donors) != 1:
            continue
        d = donors[0]
        hd = np.array(xyz[d]) - np.array(xyz[i])
        nd = float(np.linalg.norm(hd))
        for k, b in enumerate(atoms):
            if b[0] not in ("O", "N") or k in (i, d):
                continue
            ha = np.array(xyz[k]) - np.array(xyz[i])
            r = float(np.linalg.norm(ha))
            if r > h_max or r < 1e-9:
                continue
            ang = math.degrees(math.acos(
                max(-1.0, min(1.0, float(np.dot(hd, ha)) / (nd * r)))))
            if ang >= ang_min:
                out.append((r, ang))
    return out


def has_hbond(atoms, bonded, *, h_max: float = 2.5, ang_min: float = 120.0) -> bool:
    return count_hbonds(atoms, bonded, h_max=h_max, ang_min=ang_min) > 0


def count_hbonds(atoms, bonded, *, h_max: float = 2.5, ang_min: float = 120.0) -> int:
    """Number of intramolecular H-bonds (donor H -> O/N acceptor) in one conformer."""
    return len(hbond_contacts(atoms, bonded, h_max=h_max, ang_min=ang_min))


def cra_from_counts(ok: int, tot: int) -> float | None:
    """CRA from correct/total pair counts."""
    return (2.0 * ok / tot - 1.0) if tot else None


def equal_hb_cra(pairs, public_map: dict, ranking: list[str]) -> float | None:
    """CRA of one ranking on the given (lo, hi) conformer pairs."""
    conf_to_public = {int(index): str(label) for label, index in public_map.items()}
    pos = {label: i for i, label in enumerate(ranking)}
    ok = tot = 0
    for lo, hi in pairs:
        a = conf_to_public.get(lo)
        b = conf_to_public.get(hi)
        if a is None or b is None or a not in pos or b not in pos:
            continue
        tot += 1
        ok += pos[a] < pos[b]
    return cra_from_counts(ok, tot)


def load_r2scan() -> dict[str, dict[str, float]]:
    return json.loads(
        (_REPO / "orca_r2scan-3c" / "deltaE_r2scan3c_kcal_mol.json").read_text())


def bench_molecules() -> set[str]:
    """The 27 scored molecules, from the seed-0 bundle metadata."""
    meta_path = _REPO / "geom-qm9" / "prompt_30x30_NEW" / "prompt_metadata.jsonl"
    out = set()
    for line in meta_path.read_text().splitlines():
        if line.strip():
            out.add(str(json.loads(line)["molecule"]))
    return out


def hbond_pair_tables(r2scan, counts) -> dict:
    """Truth-ordered pairs (lo, hi) by H-bond count: noHB/bothHB/fav/dec.

    ``fav``/``dec`` include every pair with a different number of H-bonds;
    the side with more H-bonds is used for the conventional/decoy assignment.
    """
    tables: dict[str, dict] = {k: defaultdict(list)
                               for k in ("noHB", "bothHB", "fav", "dec")}
    for mol, cnt in counts.items():
        for a, b in itertools.combinations(sorted(r2scan[mol], key=int), 2):
            da, db = float(r2scan[mol][a]), float(r2scan[mol][b])
            if abs(da - db) < FLOOR:
                continue
            na, nb = cnt[a], cnt[b]
            lo, hi = (int(a), int(b)) if da < db else (int(b), int(a))
            if na == nb:
                tables["noHB" if na == 0 else "bothHB"][mol].append((lo, hi))
            else:
                h = a if na > nb else b
                tables["fav" if int(h) == lo else "dec"][mol].append((lo, hi))
    return tables


def hb_count_map(r2scan, bench) -> dict:
    """{mol: {conf_idx(str): n_H-bonds}} per conformer."""
    conf = geom_conformers_dir()
    out = {}
    for mol in r2scan:
        if mol not in bench:
            continue
        out[mol] = {}
        for ci in r2scan[mol]:
            atoms = read_xyz(conf / mol / f"{mol}_conf_{ci}.xyz")
            out[mol][ci] = count_hbonds(atoms, bonded_graph(atoms))
    return out


def equal_hbond_pair_tables(r2scan, counts) -> dict:
    """Truth-ordered pairs whose two conformers carry the SAME number of H-bonds.

    The H-bond heuristic gives no signal on these pairs (0-vs-0, 1-vs-1, ...).
    """
    table: dict[str, list] = defaultdict(list)
    for mol, cnt in counts.items():
        for a, b in itertools.combinations(sorted(r2scan[mol], key=int), 2):
            if abs(float(r2scan[mol][a]) - float(r2scan[mol][b])) < FLOOR:
                continue
            if cnt[a] != cnt[b]:
                continue
            lo, hi = (int(a), int(b)) if float(r2scan[mol][a]) < float(r2scan[mol][b]) \
                else (int(b), int(a))
            table[mol].append((lo, hi))
    return dict(table)


def load_meta(bundle_dir: Path, override) -> dict:
    return augment_meta(load_metadata_index(bundle_dir / "prompt_metadata.jsonl"),
                        override, strict=True)


def load_predictions(bundle_dir: Path, pred_file: str) -> dict[str, list[str]]:
    ans = {}
    for line in (bundle_dir / pred_file).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            if isinstance(r.get("prediction"), list):
                ans[str(r["molecule"])] = [str(x) for x in r["prediction"]]
    return ans


def score_pairs_letters(bundle_dirs, pred_file, tables, override) -> dict:
    """{kind: {mol: [ok, tot]}} pooled over seeds, per molecule."""
    out = {kind: defaultdict(lambda: [0, 0]) for kind in tables}
    for bdir in bundle_dirs:
        meta = load_meta(bdir, override)
        ans = load_predictions(bdir, pred_file)
        for kind, table in tables.items():
            for mol, ps in table.items():
                if mol not in ans or mol not in meta:
                    continue
                l2c = {lab: int(c) for lab, c in
                       meta[mol]["public_label_to_conf_index"].items()}
                rk = ans[mol]
                if set(rk) != set(l2c):
                    continue
                pos = {lab: i for i, lab in enumerate(rk)}
                c2l = {c: lab for lab, c in l2c.items()}
                st = out[kind][mol]
                for lo, hi in ps:
                    if lo not in c2l or hi not in c2l:
                        continue
                    st[1] += 1
                    st[0] += pos[c2l[lo]] < pos[c2l[hi]]
    return out


def score_pairs_ff(getter, tables, r2scan) -> dict:
    """{kind: {mol: [ok, tot]}} for one deterministic force-field ranking."""
    conf = geom_conformers_dir()
    out = {kind: {} for kind in tables}
    for kind, table in tables.items():
        acc = out[kind]
        for mol, ps in table.items():
            rk = getter(conf / mol)
            if not rk:
                continue
            rk = [int(c) for c in rk]
            pos = {c: i for i, c in enumerate(rk)}
            ok = tot = 0
            for lo, hi in ps:
                if lo not in pos or hi not in pos:
                    continue
                tot += 1
                ok += pos[lo] < pos[hi]
            if tot:
                acc[mol] = [ok, tot]
    return out


def pooled(per_mol: dict) -> tuple[int, int]:
    ok = sum(v[0] for v in per_mol.values())
    tot = sum(v[1] for v in per_mol.values())
    return ok, tot


def override_r2scan3c():
    return load_deltae_override(resolve_deltae_json("r2scan3c", None))

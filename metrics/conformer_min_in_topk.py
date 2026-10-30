#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from metrics.kendall_tau_ranking import load_metadata_index


def letters_at_global_minimum(letter_to_de: dict[str, float]) -> set[str]:
    m = min(letter_to_de.values())
    return {L for L, e in letter_to_de.items() if e == m}


def min_in_topk(ranking: list[str], min_letters: set[str], k: int) -> bool:
    k_eff = min(k, len(ranking))
    top = set(ranking[:k_eff])
    return bool(min_letters & top)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Hit rate: global ΔE minimum in predicted top-k."
    )
    ap.add_argument("--metadata-jsonl", type=Path, required=True)
    ap.add_argument("--answers-jsonl", type=Path, required=True)
    ap.add_argument(
        "--k",
        type=int,
        nargs="+",
        required=True,
        metavar="K",
        help="One or more k values (e.g. 1 3 5)",
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        metavar="FILE",
        help="Write report to file (UTF-8)",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-molecule lines (min letter(s), hit/miss per k)",
    )
    ap.add_argument(
        "--include-degenerate",
        action="store_true",
        help="Include excluded template rankings in hit rates (default: exclude)",
    )
    ap.add_argument(
        "--include-trivial",
        action="store_true",
        help="Alias for --include-degenerate",
    )
    ap.add_argument(
        "--full",
        action="store_true",
        help="Denominator = all molecules in metadata; failures count as misses (see docstring)",
    )
    args = ap.parse_args()
    include_degenerate = args.include_degenerate or args.include_trivial
    from metrics.ranking_validity import degenerate_pattern_reason

    meta = load_metadata_index(args.metadata_jsonl)
    ks = sorted(set(args.k))

    rows: list[tuple[str, dict[str, float], list[str]]] = []
    skipped_lines = 0

    for lineno, line in enumerate(
        args.answers_jsonl.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        raw = line.strip()
        if not raw:
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            skipped_lines += 1
            print(f"Warning: skip answers line {lineno}: invalid JSON ({e})", file=sys.stderr)
            continue
        mol = str(obj.get("molecule", ""))
        rnk = obj.get("ranking")
        if not mol:
            skipped_lines += 1
            print(f"Warning: skip answers line {lineno}: no molecule field", file=sys.stderr)
            continue
        if not isinstance(rnk, list):
            skipped_lines += 1
            print(
                f"Warning: skip answers line {lineno} ({mol!r}): ranking is not a list",
                file=sys.stderr,
            )
            continue
        ranking = [str(x) for x in rnk]
        if mol not in meta:
            skipped_lines += 1
            print(
                f"Warning: skip answers line {lineno} ({mol!r}): no such molecule in metadata",
                file=sys.stderr,
            )
            continue
        letter_de: dict[str, float] = meta[mol]["letter_to_deltaE_kcal_mol"]
        keys = set(letter_de.keys())
        rset = set(ranking)
        if rset != keys:
            miss = sorted(keys - rset, key=lambda x: (int(x) if str(x).isdigit() else x))
            extra = sorted(rset - keys, key=lambda x: (int(x) if str(x).isdigit() else x))
            skipped_lines += 1
            print(
                f"Warning: skip answers line {lineno} ({mol!r}): not a permutation of metadata labels "
                f"(missing {miss or '—'}, extra {extra or '—'}, len {len(ranking)} vs {len(keys)})",
                file=sys.stderr,
            )
            continue
        rows.append((mol, letter_de, ranking))

    valid_by_mol: dict[str, tuple[dict[str, float], list[str]]] = {}
    for mol, letter_de, ranking in rows:
        valid_by_mol[mol] = (letter_de, ranking)
    rows = [(m, t[0], t[1]) for m, t in sorted(valid_by_mol.items())]

    degenerate_count = 0
    for _mol, letter_de, ranking in rows:
        keys_i = set(letter_de.keys())
        if degenerate_pattern_reason(ranking, keys_i) is not None:
            degenerate_count += 1

    hits: dict[int, int] = {k: 0 for k in ks}
    out_lines: list[str] = []
    counted = 0

    if args.full:
        n_meta = len(meta)
        for mol in sorted(meta.keys()):
            letter_de = meta[mol]["letter_to_deltaE_kcal_mol"]
            min_L = letters_at_global_minimum(letter_de)
            keys_i = set(letter_de.keys())

            if mol not in valid_by_mol:
                if args.verbose:
                    parts = [
                        mol,
                        f"min_letter={','.join(sorted(min_L))}",
                        "pred_first=—",
                    ]
                    for kk in ks:
                        parts.append(f"@{kk}=miss (no valid answer)")
                    out_lines.append("\t".join(parts))
                continue

            ranking = valid_by_mol[mol][1]
            deg = degenerate_pattern_reason(ranking, keys_i)
            skip = deg is not None and not include_degenerate

            if args.verbose:
                min_str = ",".join(sorted(min_L))
                parts = [mol, f"min_letter={min_str}", f"pred_first={ranking[0]!r}"]
                if skip:
                    for kk in ks:
                        parts.append(f"@{kk}=miss ({deg})")
                else:
                    for kk in ks:
                        ok = min_in_topk(ranking, min_L, kk)
                        parts.append(f"@{kk}={'hit' if ok else 'miss'}")
                out_lines.append("\t".join(parts))

            if skip:
                continue
            for kk in ks:
                if min_in_topk(ranking, min_L, kk):
                    hits[kk] += 1
        denom = n_meta
    else:
        for mol, letter_de, ranking in rows:
            min_L = letters_at_global_minimum(letter_de)
            keys_i = set(letter_de.keys())
            deg = degenerate_pattern_reason(ranking, keys_i)
            skip = deg is not None and not include_degenerate

            if args.verbose:
                min_str = ",".join(sorted(min_L))
                parts = [mol, f"min_letter={min_str}", f"pred_first={ranking[0]!r}"]
                if skip:
                    for kk in ks:
                        parts.append(f"@{kk}=excluded ({deg})")
                else:
                    for kk in ks:
                        ok = min_in_topk(ranking, min_L, kk)
                        parts.append(f"@{kk}={'hit' if ok else 'miss'}")
                out_lines.append("\t".join(parts))

            if skip:
                continue
            counted += 1
            for kk in ks:
                if min_in_topk(ranking, min_L, kk):
                    hits[kk] += 1
        denom = counted if counted else (len(rows) if include_degenerate else 0)

    summary: list[str] = []
    for kk in ks:
        h = hits[kk]
        pct = 100.0 * h / denom if denom else 0.0
        summary.append(f"k={kk}: {h}/{denom}  ({pct:.3f}%)")

    header_note: list[str] = []
    if args.full:
        header_note.append(
            f"Full denominator (--full): {denom} molecule(s) in metadata; "
            "missing/invalid permutations and excluded templates (unless --include-degenerate) "
            "count as misses."
        )
    if skipped_lines:
        header_note.append(
            f"Skipped {skipped_lines} answer line(s) (bad JSON, unknown molecule, or invalid permutation)."
        )
    if not include_degenerate and degenerate_count:
        if args.full:
            header_note.append(
                f"{degenerate_count} excluded template ranking(s) scored as miss at each k."
            )
        else:
            header_note.append(
                f"Excluded {degenerate_count} template ranking(s); counted {denom} molecule(s)."
            )
    elif include_degenerate:
        header_note.append(f"Included all {len(rows)} molecule(s) (--include-degenerate).")
    if denom == 0 and rows and not args.full:
        header_note.append(
            "No molecules counted after exclusions; use --include-degenerate or fix rankings."
        )

    text = ""
    if args.verbose and out_lines:
        text += "\n".join(out_lines) + "\n\n"
    if header_note:
        text += "\n".join(header_note) + "\n"
    text += "\n".join(summary) + "\n"

    print(text, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()

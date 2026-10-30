#!/usr/bin/env python3
"""Reproduce the geom-qm9-neutral prompt bundles from the geom-qm9 paper bundles.

A neutral prompt is the paper prompt with the feature hints stripped out.
``prompt_metadata.jsonl`` is copied untouched and every ``*.txt`` gets exactly
the three replacements below, so the conformer blocks stay byte-identical to
the paper bundles and the old-vs-neutral comparison is paired.

    python prompts/make_neutral_prompts.py
    python prompts/make_neutral_prompts.py --src geom-qm9 --dst /tmp/neutral_check
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent

BUNDLES = ["prompt_30x30_NEW", "prompt_30x30_NEW_43", "prompt_30x30_NEW_44"]

REPLACEMENTS = [
    (
        (
            "coordinates (e.g. ring strain, eclipsing interactions, close "
            "nonbonded contacts, hydrogen-bond donor/acceptor proximity, "
            "dipole alignment)."
        ),
        "coordinates.",
    ),
    (
        "(step-by-step analysis of geometry, strain, contacts, uncertainty)",
        "(step-by-step analysis of the geometry)",
    ),
    (
        (
            '"reasoning": "Analyze hydrogen bonds, steric clashes, ring '
            'strain, etc. step-by-step BEFORE committing to the final '
            'order."'
        ),
        (
            '"reasoning": "Analyze the structures step-by-step BEFORE '
            'committing to the final order."'
        ),
    ),
]


def strip_text(text: str, path: Path) -> str:
    for old, new in REPLACEMENTS:
        if old not in text:
            raise SystemExit(f"Pattern missing in {path}: {old[:60]!r}...")
        text = text.replace(old, new)
    leftovers = [
        t for t in ("ring strain", "hydrogen bond", "steric clash", "dipole",
                    "transannular", "eclips")
        if t in text
    ]
    if leftovers:
        raise SystemExit(f"Hints left in {path}: {leftovers}")
    return text


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path, default=_REPO / "geom-qm9")
    ap.add_argument("--dst", type=Path, default=_REPO / "geom-qm9-neutral")
    args = ap.parse_args()

    if not args.src.is_dir():
        raise SystemExit(f"Not a directory: {args.src}")

    for bundle in BUNDLES:
        src_dir = args.src / bundle
        dst_dir = args.dst / bundle
        if not src_dir.is_dir():
            raise SystemExit(f"Missing source bundle: {src_dir}")
        dst_dir.mkdir(parents=True, exist_ok=True)

        meta = src_dir / "prompt_metadata.jsonl"
        if not meta.is_file():
            raise SystemExit(f"Missing {meta}")
        shutil.copyfile(meta, dst_dir / "prompt_metadata.jsonl")

        n = 0
        for txt in sorted(src_dir.glob("*.txt")):
            (dst_dir / txt.name).write_text(
                strip_text(txt.read_text(encoding="utf-8"), txt),
                encoding="utf-8",
            )
            n += 1
        print(f"{bundle}: {n} prompts", flush=True)


if __name__ == "__main__":
    sys.exit(main())

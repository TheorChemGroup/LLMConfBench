"""Model tables for the manuscript, generated from ``models.yaml``.

Writes the two published tables, both ordered by ``table_rank``:

* ``figures/models_table.tsv`` — display name, exact identifier, developer and
  the released / accessed dates;
* ``figures/models_reasoning_params.tsv`` — reasoning mode and parameter count
  as printed in the paper.

    python viz_helper/cohort_tables.py
"""
from pathlib import Path
import sys

import yaml

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"
sys.path[:0] = [str(REPO), str(Path(__file__).parent)]

from config.models_config import MODELS_YAML  # noqa: E402

COHORT = "figures/models_table.tsv"
PARAMS = "figures/models_reasoning_params.tsv"


def _ranked_models() -> list[dict]:
    models = yaml.safe_load(MODELS_YAML.read_text(encoding="utf-8"))["models"]
    rows = [m for m in models.values() if m.get("table_rank")]
    return sorted(rows, key=lambda m: m["table_rank"])


def main() -> list[Path]:
    rows = _ranked_models()
    OUT.mkdir(exist_ok=True)

    cohort = OUT / "models_table.tsv"
    with cohort.open("w", encoding="utf-8") as fh:
        fh.write("Model\tExact identifier\tDeveloper / provider\tReleased / accessed\n")
        for m in rows:
            fh.write(f"{m['display']}\t{m['identifier']}\t{m['provider']}\t"
                     f"{m['accessed']}\n")

    params = OUT / "models_reasoning_params.tsv"
    with params.open("w", encoding="utf-8") as fh:
        fh.write("Model\tReasoning\tParameters\n")
        for m in rows:
            fh.write(f"{m['display']}\t{m['reasoning_note']}\t{m['params_note']}\n")

    return [cohort, params]


if __name__ == "__main__":
    for path in main():
        print(f"Saved {path}")

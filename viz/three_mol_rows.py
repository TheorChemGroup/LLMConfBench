#!/usr/bin/env python3


from __future__ import annotations

import os
import re
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw as RDDraw
from rdkit.Chem.Draw import rdMolDraw2D

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

WIDTH_PX = 2126
HEIGHT_PX = 406
N_ROWS = 3
N_COLS = 9
N_TOTAL = N_ROWS * N_COLS
GAP = 0.06

BOND_PT = 10.8
BOND_WIDTH_PT = 1.8
BOND_SPACING = 0.18
MARGIN_PT = 1.2
FONT_PT = 20


def _resolve_font() -> Path:
    """First available sans-serif font (override with MOL_FONT); RDKit's FreeSans as last resort."""
    candidates = (
        os.environ.get("MOL_FONT"),
        "/usr/share/fonts/urw-base35/NimbusSans-Regular.otf",
        "/usr/share/fonts/type1/urw-base35/NimbusSans-Regular.otf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return Path(RDDraw.__file__).resolve().parent / "FreeSans.ttf"


FONT_FILE = str(_resolve_font())
OUT_PNG = Path("figures/r2scan3c/molecules_27_3row.png")
OUT_SVG = Path("figures/r2scan3c/molecules_27_3row.svg")


_PLOT_SMILES_CANONICAL = {
    "C_C_H__C_O_CO_CH__NH_": "C[C@H](C=O)COC=N",
    "C_C___NH__OCCCCO": "CC(=N)OCCCCO",
}


def mol_to_png(smi: str, width: int, height: int, fixed_bond_px: float) -> Image.Image:
    return Image.open(BytesIO(_draw(smi, width, height, fixed_bond_px, png=True))).convert("RGBA")


def mol_to_svg(smi: str, width: int, height: int, fixed_bond_px: float) -> str:
    return _draw(smi, width, height, fixed_bond_px, png=False)


def _draw(smi: str, width: int, height: int, fixed_bond_px: float, *, png: bool) -> bytes | str:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        raise ValueError(f"unparseable SMILES: {smi}")
    mol = rdMolDraw2D.PrepareMolForDrawing(mol)
    mean = rdMolDraw2D.MeanBondLength(mol)
    drawer = (
        rdMolDraw2D.MolDraw2DCairo(width, height) if png
        else rdMolDraw2D.MolDraw2DSVG(width, height)
    )
    opts = drawer.drawOptions()
    rdMolDraw2D.SetACS1996Mode(opts, mean)
    opts.bondLineWidth = BOND_WIDTH_PT
    opts.scaleBondWidth = False
    opts.fixedBondLength = fixed_bond_px
    opts.multipleBondOffset = BOND_SPACING
    opts.fixedFontSize = FONT_PT
    opts.minFontSize = FONT_PT
    opts.maxFontSize = FONT_PT
    opts.additionalAtomLabelPadding = MARGIN_PT / FONT_PT
    opts.clearBackground = False
    opts.padding = 0.08
    opts.fontFile = FONT_FILE
    opts.singleColourWedgeBonds = True
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def nest(svg: str, x: float, y: float, w: float, h: float) -> str:
    svg = re.sub(r"<\?xml[^>]*\?>", "", svg).strip()
    svg = re.sub(r"<!-- END OF HEADER -->", "", svg)
    svg = re.sub(r"""\s+width=['"][^'"]*['"]""", "", svg, count=1)
    svg = re.sub(r"""\s+height=['"][^'"]*['"]""", "", svg, count=1)
    return re.sub(
        r"<svg\b",
        f'<svg x="{x}" y="{y}" width="{w}" height="{h}"',
        svg,
        count=1,
    )


MOL_TABLE = Path("figures/r2scan3c/seed_spread_heatmap_molecules.tsv")


def load_heatmap_order(table: Path) -> tuple[list[str], dict[str, str]]:
    """(molecule ids in heatmap rank order, smiles map) from the seed-spread TSV."""
    mols: list[str] = []
    smiles_map: dict[str, str] = {}
    lines = table.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.split("\t")
        mol_id = parts[1]
        mols.append(mol_id)
        smi = parts[2]
        if smi:
            smiles_map[mol_id] = smi
    return mols, smiles_map


def main() -> Path:
    if not MOL_TABLE.is_file():
        raise SystemExit(f"missing {MOL_TABLE}; run viz_helper/seed_spread_heatmap.py first")
    molecules, smiles_map = load_heatmap_order(MOL_TABLE)
    if len(molecules) != N_TOTAL:
        raise SystemExit(f"expected {N_TOTAL} molecules in {MOL_TABLE}, got {len(molecules)}")
    missing = [m for m in molecules if not smiles_map.get(m)]
    if missing:
        raise SystemExit(f"no SMILES for: {missing}")

    cell_w = WIDTH_PX / N_COLS
    cell_h = HEIGHT_PX / N_ROWS
    draw_w = max(1, (round(cell_w * (1.0 - GAP))))
    draw_h = max(1, (round(cell_h * (1.0 - GAP))))
    pad_x = (cell_w - draw_w) / 2.0
    pad_y = (cell_h - draw_h) / 2.0

    fixed_bond_px = BOND_PT * draw_h / 60.0

    canvas = Image.new("RGBA", (WIDTH_PX, HEIGHT_PX), (255, 255, 255, 0))
    svgs: list[str] = []
    for i, mol in enumerate(molecules):
        smi = smiles_map[mol]
        plot_smi = _PLOT_SMILES_CANONICAL.get(mol, smi)
        r, c = divmod(i, N_COLS)
        x = round(c * cell_w + pad_x)
        y = round(r * cell_h + pad_y)
        cell = mol_to_png(plot_smi, draw_w, draw_h, fixed_bond_px)
        canvas.paste(cell, (x, y), cell)
        svgs.append(nest(mol_to_svg(plot_smi, draw_w, draw_h, fixed_bond_px), x, y, draw_w, draw_h))
        if plot_smi != smi:
            print(mol, smi, "->", plot_smi)
        else:
            print(mol, smi)

    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PNG)
    OUT_SVG.write_text(
        "\n".join([
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH_PX}" '
                f'height="{HEIGHT_PX}" viewBox="0 0 {WIDTH_PX} {HEIGHT_PX}">'
            ),
            *svgs,
            "</svg>",
        ]),
        encoding="utf-8",
    )
    print(
        f"Saved {OUT_PNG} and {OUT_SVG}  ({WIDTH_PX}×{HEIGHT_PX} px, "
        f"cell {draw_w}×{draw_h}, bond {fixed_bond_px:.1f} px; "
        f"{len(molecules)} molecules in seed-spread heatmap order)"
    )
    return OUT_PNG


if __name__ == "__main__":
    main()

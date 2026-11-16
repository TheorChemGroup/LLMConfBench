"""Render the conformer images for the Fig. 3a gallery with xyzrender.

Requires an environment providing ``xyzrender`` and ``cairosvg`` (override with
``XYZRENDER`` and ``CAIROSVG_PYTHON``). Rendered PNGs are read by
``decoy_gallery.py``, which ``viz/first_plots.py`` calls.

    python viz_helper/render_decoy_gallery.py
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parent.parent
OUT = Path(os.environ.get("FIG3_GALLERY_OUT", REPO / "figures" / "fig3_gallery"))
CONF = REPO / "geom-qm9" / "output_30_confs"


def _xyzrender() -> str:
    if os.environ.get("XYZRENDER"):
        return os.environ["XYZRENDER"]
    local = Path(sys.executable).with_name("xyzrender")
    return str(local) if local.is_file() else "xyzrender"


XYZR = _xyzrender()
PY = os.environ.get("CAIROSVG_PYTHON", sys.executable)

# Atom indices are 1-based, as used by the xyzrender CLI.
MOLS = {
    4:  dict(id="O_CNC_C_H__O__C_H_1CO1", low=4, hb=21, h=2, donor=1, acc=7),
    8:  dict(id="O_CC_C__H_1CO_C__H_1CO", low=10, hb=24, h=2, donor=1, acc=9),
    16: dict(id="C_C__H__CO_CCNC_O", low=4, hb=27, h=20, donor=19, acc=10),
    21: dict(id="OCC_C_H_1C_C_H_2OC_C__H_12", low=0, hb=23, h=2, donor=1, acc=12),
    22: dict(id="CC__O_C_C___1_O_C_C_H_1O", low=8, hb=9, h=11, donor=10, acc=3),
}

BASE = ["-S", "1000", "-a", "1.2", "-b", "5", "-s", "0.6",
        "--hy", "--h-scale", "0.8", "-B", "white",
        "--fog", "-F", "0.15", "--bond-color", "#555555"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for mol, s in MOLS.items():
        mid = s["id"]
        ref = OUT / f"ref{mol}.xyz"
        for tag, conf, hb in (("hb", s["hb"], True), ("low", s["low"], False)):
            svg = OUT / f"mol{mol}_{tag}.svg"
            cmd = [XYZR, str(CONF / mid / f"{mid}_conf_{conf}.xyz"),
                   "-o", str(svg), *BASE]
            if hb:
                cmd += ["--radius-scale", str(s["h"]), "1.6",
                        "--nci-bond", f"{s['h']}-{s['acc']}",
                        "--nci-color", "#333333", "--nci-dash", "3,2",
                        "--nci-width", "1.0"]
            if ref.is_file():
                cmd += ["--ref", str(ref)]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL)

            png = OUT / f"mol{mol}_{tag}.png"
            code = (f"import cairosvg; cairosvg.svg2png(url={str(svg)!r}, "
                    f"write_to={str(png)!r}, output_width=1000, "
                    f"output_height=1000, background_color='white')")
            subprocess.run([PY, "-c", code], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"Rendered {len(MOLS)} molecules into {OUT}")


if __name__ == "__main__":
    main()

"""Fig. 3b quotes: seven model excerpts with outcome lines, two-column panel.

Each block shows a verbatim excerpt (elisions marked ``<...>``), the molecule and
model it came from, and what the ranking actually did compared with the
r2SCAN-3c energies. Bracketed conformer labels that are our insertion rather
than the model's text are set in italic at half opacity.

    python viz_helper/quotes_panel.py
"""
from pathlib import Path
import json
import re
import textwrap

from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "figures"

NCOL = 2
W = 4000
MARGIN = 120
GUTTER = 170
COL_W = (W - 2 * MARGIN - GUTTER * (NCOL - 1)) // NCOL
X0, XR = MARGIN, MARGIN + COL_W
CHARS_SCALE = COL_W / 2260.0
FONT_DIR = Path("/usr/share/fonts/urw-base35")
REG = str(FONT_DIR / "NimbusSans-Regular.otf")
BOLD = str(FONT_DIR / "NimbusSans-Bold.otf")
ITAL = str(FONT_DIR / "NimbusSans-Italic.otf")

S_MODEL, S_QUOTE, S_OUT = 10.0, 9.0, 7.6
F_MODEL = ImageFont.truetype(BOLD, round(S_MODEL * 500 / 72))
F_QUOTE = ImageFont.truetype(REG, round(S_QUOTE * 500 / 72))
F_QUOTE_I = ImageFont.truetype(ITAL, round(S_QUOTE * 500 / 72))
F_OUT = ImageFont.truetype(REG, round(S_OUT * 500 / 72))
F_OUT_B = ImageFont.truetype(BOLD, round(S_OUT * 500 / 72))

CHARS = 63
GAP = 55

BLOCKS = [
    dict(model="GPT-6 Astra", mol="mol 16", color="#c62828", nums={9, 13, 2}, chars=68,
         quote="“<...> Hydrogen-bond distance alone would overrate some structures. Although 9, 13, "
               "and 2 have short OH···O=C contacts, their hydroxymethyl torsions move toward "
               "unfavorable eclipsing regions; 13 and 2 also require appreciable backbone "
               "twisting. <...>”",
         outcome="13 and 2 were placed 15th and 16th; they are 29th and 30th of 30 by energy."),
    dict(model="Claude Opus 5", mol="mol 16", color="#c62828", nums={10, 11, 18, 19, 6},
         quote="“<...> the relative weight of the strong-but-strained OH···O=C macrocycle versus "
               "the strain-free NH···O 7-ring is the main ambiguity, so the boundary region "
               "(10, 11 vs 18, 19, 6) could easily reorder <...>”",
         outcome="10 and 11 were placed 9th and 10th, above 18, 19 and 6; they are 22nd and 24th of 30 by energy."),
    dict(model="Claude Opus 5", mol="mol 21", color="#ef6c00", nums={14, 9}, chars=68,
         quote="“<...> These two (14, 9) are clearly the most stable; 14 has the more linear, less "
               "squeezed bond <...> so 14 is placed first <...>”",
         ital={"(14,", "9)"},
         outcome="14 and 9 were placed 1st and 2nd; they are 19th and 12th of 30 by energy."),
    dict(model="Claude Opus 5", mol="mol 8", color="#1565c0", nums={5, 22, 13},
         quote="“<...> These three (5, 22, 13) are the only ones with a genuine carbonyl "
               "hydrogen bond and are put first <...> they trade the weak ring-oxygen "
               "contact for a much stronger carbonyl bond — a net gain of a few "
               "kcal/mol. <...>”",
         ital={"(5,", "22,", "13)"},
         outcome="22, 5 and 13 took the top three places; 13 is 19th of 30 by energy, 5 is 3rd."),
    dict(model="Kimi K3", mol="mol 4", color="#2e7d32", nums={16},
         quote="“<...> Conformer 16 has a short O–H···O7 contact near 2.00 Å and a 120-degree "
               "donor angle <...> Alcohol O–H donation makes 16 slightly preferable overall.”",
         outcome="16 was placed 5th; it is 30th of 30 by energy."),
    dict(model="GPT-6 Astra", mol="mol 4", color="#2e7d32", nums={16},
         quote="“<...> Torsional strain prevents ranking by hydrogen-bond distance alone. "
               "Structure 16 has an attractive OH···O(epoxide) contact, approximately "
               "2.00 angstrom/122 degrees, but its alcohol-O–CH–CH2–N torsion is nearly "
               "eclipsed, approximately 7 degrees <...> I therefore demote it substantially.”",
         outcome="16 was placed 21st; it is 30th of 30 by energy."),
    dict(model="GPT-5.6 Sol", mol="mol 22", color="#6a1b9a", nums={3, 9, 21},
         quote="“<...> The strongest six-membered intramolecular O–H···O=C arrangements occur "
               "in 3 and 9 (H···O about 1.87 Å, angle about 157°) and 21 (1.91 Å, about 158°). "
               "Conformer 3 is placed just ahead of 9 <...>”",
         outcome="3, 9 and 21 were placed 1st–3rd; they are 21st, 16th and 12th of 30 by energy."),
]


# One verdict per block, in the order of BLOCKS.
VERDICTS = ("wrong", "wrong", "wrong", "wrong", "wrong", "partly corrected", "wrong")


def token_color(tok, nums, color):
    core = re.sub(r"^[\s“”\"'()\[\]<>]+|[\s“”\"'().,;:\[\]<>]+$", "", tok)
    if core.isdigit() and int(core) in nums:
        return color
    return "black"


def half(color):
    if color == "black":
        color = (0, 0, 0)
    elif isinstance(color, str):
        color = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    return tuple(round((c + 255) / 2) for c in color)


def lh(s_pt, mult):
    return round(s_pt * 500 / 72 * mult)


def verdict(b) -> str:
    return VERDICTS[BLOCKS.index(b)]


def lines_of(b):
    width = round(b.get("chars", CHARS) * CHARS_SCALE)
    q = textwrap.wrap(b["quote"], width=width)
    prefix = 2 + len(verdict(b)) + 2
    o = textwrap.wrap(b["outcome"], width=round((88 - prefix) * CHARS_SCALE))
    return q, o


def block_height(b):
    q, o = lines_of(b)
    return (lh(S_MODEL, 1.25) + 10 + len(q) * lh(S_QUOTE, 1.20) + GAP
            + len(o) * lh(S_OUT, 1.25))


def draw_block(img, top, b, x0=None, xr=None):
    x0 = X0 if x0 is None else x0
    xr = XR if xr is None else xr
    draw = ImageDraw.Draw(img)
    x_right = xr
    y = top
    draw.text((x0, y), b["model"], font=F_MODEL, fill="black")
    tw = draw.textlength(b["mol"], font=F_MODEL)
    draw.text((x_right - tw, y), b["mol"], font=F_MODEL, fill=b["color"])
    y += lh(S_MODEL, 1.25) + 10

    q, o = lines_of(b)
    for line in q:
        x = x0
        for tok in line.split(" "):
            col = token_color(tok, b["nums"], b["color"])
            if tok in b.get("ital", ()):
                f, col = F_QUOTE_I, half(col)
            else:
                f = F_QUOTE
            draw.text((x, y), tok, font=f, fill=col)
            x += draw.textlength(tok, font=f) + draw.textlength(" ", font=F_QUOTE)
        y += lh(S_QUOTE, 1.20)
    y += GAP

    lab = verdict(b)
    for i, line in enumerate(o):
        if i == 0:
            x = x0
            draw.text((x, y), "→ ", font=F_OUT, fill="#333333")
            x += draw.textlength("→ ", font=F_OUT)
            draw.text((x, y), lab, font=F_OUT_B, fill="#333333")
            x += draw.textlength(lab, font=F_OUT_B)
            draw.text((x, y), ":", font=F_OUT, fill="#333333")
            x += draw.textlength(":", font=F_OUT) + draw.textlength(" ", font=F_OUT)
            draw.text((x, y), line, font=F_OUT, fill="#333333")
        else:
            draw.text((x0, y), line, font=F_OUT, fill="#333333")
        y += lh(S_OUT, 1.25)

    draw.rectangle([x0 - 65, top - 8, x0 - 43, y - 22], fill=b["color"])
    return y


def main() -> Path:
    per_col = -(-len(BLOCKS) // NCOL)          # 4 + 3 for seven blocks
    columns = [BLOCKS[i:i + per_col] for i in range(0, len(BLOCKS), per_col)]
    total_h = 30 + max(sum(block_height(b) for b in col) + GAP * (len(col) - 1)
                       for col in columns) + 30

    canvas = Image.new("RGB", (W, total_h), "white")
    for ci, col in enumerate(columns):
        x0 = MARGIN + ci * (COL_W + GUTTER)
        y = 30
        for b in col:
            end = draw_block(canvas, y, b, x0=x0, xr=x0 + COL_W)
            y = end + GAP

    panel = OUT / "fig3b_quotes.png"
    canvas.save(panel, dpi=(500, 500))

    # One source per block, in the order of BLOCKS.
    SOURCES = (
        ("texts/seed42/gpt6astra.jsonl", 16),
        ("texts/seed42/claudeopus5.jsonl", 16),
        ("texts/seed42/claudeopus5.jsonl", 22),
        ("texts/seed42/claudeopus5.jsonl", 12),
        ("texts/seed42/kimik3.jsonl", 11),
        ("texts/seed42/gpt6astra.jsonl", 11),
        ("texts/seed42/gpt56sol.jsonl", 15),
    )
    # Blocks whose bracketed labels are our insertion, not the model's text.
    INSERTED = {2: ("(10, 11 vs 18, 19, 6)", "verbatim"),
                3: ("(14, 9)", "expansion"),
                4: ("(5, 22, 13)", "expansion")}

    records = []
    for i, b in enumerate(BLOCKS, 1):
        src, line = SOURCES[i - 1]
        records.append({
            "id": i,
            "model": b["model"],
            "molecule": b["mol"],
            "color": b["color"],
            "conformers": sorted(b["nums"]),
            "quote": b["quote"],
            "inserted": INSERTED.get(i, ("", ""))[0],
            "inserted_type": INSERTED.get(i, ("", ""))[1],
            "outcome": b["outcome"],
            "verdict": verdict(b),
            "result": verdict(b),
            "source": {"file": src, "line": line},
        })

    with open(OUT / "fig3b_quotes.json", "w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)

    return panel


if __name__ == "__main__":
    print(f"Saved {main()}")

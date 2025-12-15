#!/usr/bin/env python3
"""Make side-by-side comparison images for LM-TAD evaluation plots.

Usage: run from repo root with Python. It will look for the following files
and write comparison images next to the originals:

- tools_eval_lmtad/Beijing/lmtad_eval_abnormality.png
- tools_eval_lmtad/Beijing_abnormal_2/lmtad_eval_abnormality.png
- tools_eval_lmtad/Beijing/lmtad_eval_boxplot.png
- tools_eval_lmtad/Beijing_abnormal_2/lmtad_eval_boxplot.png

and the same for `porto_hoser` / `porto_hoser_abnormal_2`.
"""

from pathlib import Path
from PIL import Image

PAIRS = [
    ("Beijing", "Beijing_abnormal_2"),
    ("porto_hoser", "porto_hoser_abnormal_2"),
]

FILES = [
    "lmtad_eval_abnormality.png",
    "lmtad_eval_boxplot.png",
]


def stitch(a: Path, b: Path, out: Path):
    if not a.exists() or not b.exists():
        print(f"Skipping missing pair: {a}, {b}")
        return
    ia = Image.open(a).convert("RGBA")
    ib = Image.open(b).convert("RGBA")

    # Resize to same height (preserve aspect)
    h = max(ia.height, ib.height)

    def resize_keep(img, target_h):
        w = int(img.width * (target_h / img.height))
        return img.resize((w, target_h), Image.LANCZOS)

    ia_r = resize_keep(ia, h)
    ib_r = resize_keep(ib, h)

    out_img = Image.new("RGBA", (ia_r.width + ib_r.width, h), (255, 255, 255, 255))
    out_img.paste(ia_r, (0, 0), ia_r)
    out_img.paste(ib_r, (ia_r.width, 0), ib_r)
    out.parent.mkdir(parents=True, exist_ok=True)
    out_img.save(out)
    print(f"Wrote {out}")


def main():
    root = Path("tools_eval_lmtad")
    for left, right in PAIRS:
        for fname in FILES:
            a = root / left / fname
            b = root / right / fname
            out = root / left / f"comparison_{fname}"
            stitch(a, b, out)


if __name__ == "__main__":
    main()

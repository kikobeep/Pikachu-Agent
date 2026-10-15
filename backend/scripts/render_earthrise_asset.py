"""Render https://ascii.rest/earthrise/ as a full-resolution PNG.

Node is needed only to regenerate the scene; the CLI keeps status as terminal
text over the bundled PNG. The source and MIT license live in
``app/interfaces/cli/assets``.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw


ASSETS = Path(__file__).resolve().parents[1] / "app/interfaces/cli/assets"
SOURCE = ASSETS / "earthrise_source.mjs"
OUTPUT = ASSETS / "earthrise.png"
CELL = 8

FRAME_JS = """
import { pathToFileURL } from 'node:url';
const { default: earthrise, meta } = await import(pathToFileURL(process.argv[1]).href);
const color = new Uint8Array(meta.cols * meta.rows);
const characters = earthrise()(0, { color }).replaceAll('\\n', '');
process.stdout.write(JSON.stringify({
  width: meta.cols, height: meta.rows, ground: meta.ground,
  palette: meta.palette, characters, color: Array.from(color),
}));
"""


def main() -> None:
    frame = json.loads(
        subprocess.check_output(
            ["node", "--input-type=module", "-e", FRAME_JS, str(SOURCE)],
            text=True,
        )
    )
    width, height = frame["width"], frame["height"]
    if len(frame["characters"]) != width * height or len(frame["color"]) != width * height:
        raise ValueError("earthrise frame has an unexpected size")

    image = Image.new("RGB", (width * CELL, height * CELL), frame["ground"])
    draw = ImageDraw.Draw(image)
    radii = {" ": 0, "·": 0.9, "•": 2.1, "●": 3.5}
    for index, (character, color_index) in enumerate(
        zip(frame["characters"], frame["color"], strict=True)
    ):
        radius = radii[character]
        if not radius:
            continue
        x = (index % width + 0.5) * CELL
        y = (index // width + 0.5) * CELL
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=frame["palette"][color_index],
        )
    image.save(OUTPUT, optimize=True)
    print(f"Wrote {OUTPUT} ({image.width}x{image.height})")


if __name__ == "__main__":
    main()

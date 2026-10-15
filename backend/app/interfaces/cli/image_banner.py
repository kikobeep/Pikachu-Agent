"""Show the Earthrise scene behind live CLI text in iTerm2."""

from __future__ import annotations

import base64
import io
import os
import sys
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


BACKGROUND = Path(__file__).with_name("assets") / "earthrise.png"
TRAINER_FILES = ("trainer_ash.png", "trainer_bubble.png")
PANEL_BACKGROUND = (21, 29, 40)
_background_active = False


def image_protocol(env: Mapping[str, str] | None = None) -> str | None:
    values = os.environ if env is None else env
    if values.get("PIKACHU_BANNER") == "ascii" or values.get("TMUX") or values.get("STY"):
        return None
    program = values.get("TERM_PROGRAM", "").lower()
    if program == "iterm.app" or values.get("LC_TERMINAL") == "iTerm2":
        return "iterm"
    return None


def _background_sequence(path: Path | None) -> str:
    """Set a session background image; an empty filename removes it."""
    encoded = "" if path is None else base64.b64encode(os.fsencode(path.resolve())).decode("ascii")
    return f"\x1b]1337;SetBackgroundImageFile={encoded}\x07"


@lru_cache(maxsize=12)
def _inline_image_sequence(
    path: Path, *, width: int, height: int,
    background: tuple[int, int, int] | None = None,
) -> str:
    """Paint the sprite across its entire reserved terminal rectangle."""
    if background is None:
        data = path.read_bytes()
    else:
        with Image.open(path) as source:
            sprite = source.convert("RGBA")
            canvas_size = (width * 8, height * 16)
            sprite.thumbnail(canvas_size, Image.Resampling.NEAREST)
            panel = Image.new("RGBA", canvas_size, (*background, 255))
            panel.alpha_composite(
                sprite,
                ((canvas_size[0] - sprite.width) // 2, (canvas_size[1] - sprite.height) // 2),
            )
            buffer = io.BytesIO()
            panel.convert("RGB").save(buffer, format="PNG")
            data = buffer.getvalue()
    return _inline_bytes_sequence(
        data, width=width, height=height,
        preserve_aspect_ratio=background is None,
    )


def _inline_bytes_sequence(
    data: bytes, *, width: int, height: int, preserve_aspect_ratio: bool = True,
) -> str:
    encoded = base64.b64encode(data).decode("ascii")
    return (
        f"\x1b]1337;File=inline=1;size={len(data)};width={width};height={height};"
        f"preserveAspectRatio={int(preserve_aspect_ratio)}:{encoded}\x07"
    )


@lru_cache(maxsize=16)
def _title_image_sequence(
    title: str, columns: int, *, top: bool = False,
) -> str:
    """Draw fixed headings at two cell rows while keeping the status text live."""
    background = PANEL_BACKGROUND
    foreground = (244, 239, 205) if top else (123, 232, 237)
    canvas = Image.new("RGB", (columns * 4, 16), background)
    draw = ImageDraw.Draw(canvas)
    font_path = "/System/Library/Fonts/Menlo.ttc"
    for size in range(14, 8, -1):
        try:
            font = ImageFont.truetype(font_path, size)
        except OSError:
            font = ImageFont.load_default()
        if draw.textbbox((0, 0), title, font=font)[2] <= canvas.width - 8:
            break
    draw.text((5, 0), title, font=font, fill=(8, 18, 27))
    draw.text((4, -1), title, font=font, fill=foreground)
    canvas = canvas.resize((columns * 8, 32), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return _inline_bytes_sequence(
        buffer.getvalue(), width=columns, height=2,
        preserve_aspect_ratio=False,
    )


def _portrait_overlays(frame: str, *, left: int, width: int, compact: bool) -> str:
    """Paint portraits over reserved frame cells, then restore the input cursor."""
    if width < 105:
        return ""
    from .cli_ui import _AGENT_CHOICES, _PORTRAIT_FILES, _left_column_width, _portrait_size

    image_width, _ = _portrait_size(
        compact=compact or width < 130, inline=True
    )
    image_height = 3 if compact or width < 130 else 4
    column = left + _left_column_width(width) + 8
    rows = frame.splitlines()
    line_count = len(rows)
    assets = Path(__file__).with_name("assets")
    overlays = []
    for key, name, _, _ in _AGENT_CHOICES:
        row = next(index for index, line in enumerate(rows) if name in line)
        image = _inline_image_sequence(
            assets / _PORTRAIT_FILES[key], width=image_width, height=image_height,
            background=PANEL_BACKGROUND,
        )
        overlays.append(f"\x1b7\r\x1b[{line_count - row}A\x1b[{column}C{image}\x1b8")
    return "".join(overlays)


def _trainer_overlays(
    frame: str, *, left: int, width: int, compact: bool,
    large_titles: bool = False,
) -> str:
    """Put Ash and his speech bubble in the reserved left column."""
    if width < 76:
        return ""
    from .cli_ui import _portrait_size, _trainer_art_row

    portrait_height = _portrait_size(compact=compact or width < 130, inline=True)[1]
    rows = frame.splitlines()
    chooser_rows = (2 if large_titles else 1) + (portrait_height // 2) * 4
    welcome_row = next(index for index, line in enumerate(rows) if "Welcome" in line)
    art_row = welcome_row + _trainer_art_row(chooser_rows)
    line_count = len(rows)
    if width < 130:
        positions = (
            ("trainer_ash.png", 0, 1, 13, 11),
            ("trainer_bubble.png", 12, 0, 12, 4),
        )
    elif width < 155:
        positions = (
            ("trainer_ash.png", 0, 1, 15, 11),
            ("trainer_bubble.png", 15, 0, 14, 5),
        )
    else:
        positions = (
            ("trainer_ash.png", 1, 1, 16, 11),
            ("trainer_bubble.png", 17, 0, 16, 5),
        )
    assets = Path(__file__).with_name("assets")
    overlays = []
    for filename, x, y, image_width, image_height in positions:
        row = art_row + y
        column = left + 2 + x
        image = _inline_image_sequence(
            assets / filename, width=image_width, height=image_height,
            background=PANEL_BACKGROUND,
        )
        overlays.append(f"\x1b7\r\x1b[{line_count - row}A\x1b[{column}C{image}\x1b8")
    return "".join(overlays)


def _title_overlays(frame: str, *, left: int, width: int) -> str:
    """Paint larger pixel headings into the reserved two-row title bands."""
    from .cli_ui import _chooser_width, _left_column_width

    left_width = _left_column_width(width)
    chooser_width = _chooser_width(width)
    status_width = width - left_width - chooser_width - 10
    chooser_column = left + left_width + 5
    status_column = left + left_width + chooser_width + 8
    headings = (
        ("Pikachu CLI · local agent workspace", " ▣  ▪ ▪ ▪    Pikachu CLI · local agent workspace", left + 1, width - 2, True),
        ("Welcome", "Welcome", left + 2, left_width, False),
        ("Choose your mode · 1-4", "Choose your mode · 1-4", chooser_column, chooser_width, False),
        ("▣ PIKA STATUS", "▣ PIKA STATUS", status_column, status_width, False),
        ("▣ Tips for getting started", "▣ Tips for getting started", status_column, status_width, False),
    )
    rows = frame.splitlines()
    overlays = []
    for marker, title, column, columns, top in headings:
        row = next(index for index, line in enumerate(rows) if marker in line)
        image = _title_image_sequence(title, columns, top=top)
        overlays.append(f"\x1b7\r\x1b[{len(rows) - row}A\x1b[{column}C{image}\x1b8")
    return "".join(overlays)


def print_image_banner(
    status: Sequence[tuple[str, str]], *, selected_agent: str = "1"
) -> bool:
    """Place the scalable image behind a native, independently updated frame."""
    global _background_active
    if not sys.stdout.isatty() or image_protocol() != "iterm" or not BACKGROUND.is_file():
        return False

    from .cli_ui import _PORTRAIT_FILES, _render_banner_text, terminal_size

    terminal = terminal_size((100, 30))
    if terminal.columns < 80 or terminal.lines < 24:
        return False

    compact = terminal.lines < 34
    width = min(terminal.columns - 4, max(130, round(terminal.columns * 0.86)))
    large_titles = not compact and width >= 105
    assets = Path(__file__).with_name("assets")
    inline_portraits = width >= 105 and all(
        (assets / name).is_file() for name in _PORTRAIT_FILES.values()
    )
    inline_trainer = width >= 76 and all(
        (assets / name).is_file() for name in TRAINER_FILES
    )
    frame = _render_banner_text(
        status,
        width=width,
        terminal_lines=terminal.lines,
        use_color=True,
        scene=False,
        compact=compact,
        selected_agent=selected_agent,
        inline_portraits=inline_portraits,
        inline_trainer=inline_trainer,
        large_titles=large_titles,
    )
    left = max(2, min((terminal.columns - width) // 2, terminal.columns - width))
    top = 1 if compact else 2
    prefix = "" if _background_active else _background_sequence(BACKGROUND)
    output = prefix + "\n" * top + "\n".join(" " * left + line for line in frame.splitlines()) + "\n"
    if inline_portraits:
        output += _portrait_overlays(frame, left=left, width=width, compact=compact)
    if inline_trainer:
        output += _trainer_overlays(
            frame, left=left, width=width, compact=compact,
            large_titles=large_titles,
        )
    if large_titles:
        output += _title_overlays(frame, left=left, width=width)
    sys.stdout.write(output)
    sys.stdout.flush()
    _background_active = True
    return True


def clear_image_banner() -> None:
    """Restore the terminal background when this CLI session ends."""
    global _background_active
    if _background_active and sys.stdout.isatty():
        sys.stdout.write(_background_sequence(None))
        sys.stdout.flush()
    _background_active = False

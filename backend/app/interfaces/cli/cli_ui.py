"""Pikachu CLI 展示与模型设置：直接使用 ModelConfig 和 .config 文件。"""
from __future__ import annotations

import asyncio
import getpass
import json
import os
import shutil
import sys
import tempfile
import unicodedata
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

from PIL import Image
from app.model.config import ModelConfig, ModelProvider
from app.tools.search.settings import SearchSettings
from dotenv import set_key
from pydantic import SecretStr

if TYPE_CHECKING:
    from app.agent.events import AgentEvent
    from app.checkpoint.config import RunCheckpoint
    from app.memory.regular import RegularMemoryRecord
    from app.run.config import Run
    from app.tools.permissions.models import PermissionRule


_PROVIDER_LABELS = {
    ModelProvider.OPENAI: "OpenAI",
    ModelProvider.QWEN: "Qwen",
    ModelProvider.DEEPSEEK: "DeepSeek",
}

_AGENT_CHOICES = (
    ("1", "Lucario 路卡利欧", "Aura-guided precision", "full-power coding"),
    ("2", "Jigglypuff 胖丁", "A voice full of charm", "full-power creation"),
    ("3", "Psyduck 可达鸭", "Ponders every puzzle", "plan before acting"),
    ("4", "Pikachu 皮卡丘", "Lightning-fast energy", "quick, light tasks"),
)

_MODE_BADGES = {
    "1": ("NORMAL", (24, 91, 156), (234, 246, 255)),
    "2": ("NORMAL", (24, 91, 156), (234, 246, 255)),
    "3": ("PLAN", (243, 185, 36), (28, 34, 42)),
    "4": ("LIGHT", (219, 84, 172), (255, 246, 252)),
}

_TRAINER_ART_HEIGHT = 12


def _trainer_art_row(chooser_rows: int) -> int:
    """Body row where the two supplied trainer images begin."""
    return 2 + max(0, (chooser_rows - (2 + _TRAINER_ART_HEIGHT)) // 2)

_PORTRAIT_FILES = {
    "1": "agent_lucario.png",
    "2": "agent_jigglypuff.png",
    "3": "agent_psyduck.png",
    "4": "agent_pikachu.png",
}


def _portrait_size(*, compact: bool, inline: bool) -> tuple[int, int]:
    if inline:
        return (6, 6) if compact else (8, 8)
    return (8, 6) if compact else (12, 8)


def _chooser_width(width: int) -> int:
    if width < 130:
        return 40
    return min(64, max(43, width // 3 + 8))


def _left_column_width(width: int) -> int:
    if width >= 155:
        return 34
    if width >= 130:
        return 30
    if width >= 105:
        return 24
    return 30 if width >= 76 else width - 4


@lru_cache(maxsize=12)
def _portrait_pixels(agent: str, width: int, height: int) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Render the supplied character cutouts with crisp pixel edges."""
    path = Path(__file__).with_name("assets") / _PORTRAIT_FILES[agent]
    with Image.open(path) as source:
        background = Image.new("RGBA", source.size, (21, 29, 40, 255))
        background.alpha_composite(source.convert("RGBA"))
        portrait = background.convert("RGB").resize((width, height), Image.Resampling.NEAREST)
        return tuple(
            tuple(portrait.getpixel((x, y)) for x in range(width))
            for y in range(height)
        )


def _portrait_line(agent: str, row: int, *, use_color: bool, width: int, height: int) -> str:
    pixels = _portrait_pixels(agent, width, height)
    if not use_color:
        return "▀" * width
    pieces = []
    for top, bottom in zip(pixels[row * 2], pixels[row * 2 + 1]):
        foreground = ";".join(map(str, top))
        background = ";".join(map(str, bottom))
        pieces.append(f"\033[38;2;{foreground};48;2;{background}m▀")
    return "".join(pieces) + "\033[0m"


def terminal_size(fallback: tuple[int, int] = (100, 24)) -> os.terminal_size:
    """Read the PTY size directly so stale COLUMNS/LINES cannot shrink the scene."""
    try:
        return os.get_terminal_size(sys.stdout.fileno())
    except (AttributeError, OSError, ValueError):
        return shutil.get_terminal_size(fallback)


def _terminal_display_width(text: str) -> int:
    """Return the number of terminal columns occupied by plain text."""
    width = 0
    for character in text:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


def _fit_terminal_text(text: str, width: int) -> str:
    """Truncate text by terminal columns without splitting a wide character."""
    if width <= 0:
        return ""
    result: list[str] = []
    used = 0
    for character in text:
        character_width = 0 if unicodedata.combining(character) else (
            2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        )
        if used + character_width > width:
            break
        result.append(character)
        used += character_width
    return "".join(result)


def print_banner(
    *,
    output_fn: Callable[[str], Any] = print,
    status: Sequence[tuple[str, str]] = (),
    selected_agent: str = "1",
) -> bool:
    """Render the welcome information using live terminal text."""
    if output_fn is print:
        from app.interfaces.cli.image_banner import print_image_banner

        if print_image_banner(status, selected_agent=selected_agent):
            return True

    use_color = output_fn is print and _terminal_ui_color_enabled()
    terminal = terminal_size((100, 24))
    output_fn(
        _render_banner_text(
            status,
            width=max(20, min(terminal.columns - 2, 174 if terminal.columns >= 134 else 120)),
            terminal_lines=terminal.lines,
            use_color=use_color,
            selected_agent=selected_agent,
        )
    )
    return False


def _render_banner_text(
    status: Sequence[tuple[str, str]],
    *,
    width: int,
    terminal_lines: int,
    use_color: bool,
    scene: bool = True,
    compact: bool = False,
    selected_agent: str = "1",
    inline_portraits: bool = False,
    inline_trainer: bool = False,
    large_titles: bool = False,
) -> str:
    """Build the text frame; image-capable terminals draw it over the scene."""
    split = width >= 76
    three_columns = width >= 105
    left_width = _left_column_width(width)
    chooser_width = _chooser_width(width) if three_columns else 0
    right_width = width - left_width - (chooser_width + 10 if three_columns else 7) if split else width - 4

    def style(text: str, color: str, *, background: bool = False) -> str:
        if not use_color:
            return text
        bg = ";48;2;21;29;40" if background else ""
        return f"\033[{color}{bg}m{text}\033[0m"

    def cell(text: str, cell_width: int, color: str) -> str:
        fitted = _fit_terminal_text(text, cell_width)
        return style(
            fitted + " " * (cell_width - _terminal_display_width(fitted)),
            color,
            background=True,
        )

    def heading_cell(text: str, cell_width: int, *, top: bool = False) -> str:
        fitted = _fit_terminal_text(text, cell_width)
        padded = fitted + " " * (cell_width - _terminal_display_width(fitted))
        if not use_color:
            return padded
        foreground = "244;239;205" if top else "123;232;237"
        background = "21;29;40"
        return f"\033[1;38;2;{foreground};48;2;{background}m{padded}\033[0m"

    # The supplied earthrise sketch inspires the sky and cratered foreground.
    # These are text cells so the welcome screen works in ordinary terminals.
    sky: list[str] = []
    earth = ("  ▄▓▄  ", " ▟▓▒▓▙ ", "  ▀▓▀  ")
    sky_rows = 2 if terminal_lines <= 26 else 3
    for row in range(sky_rows):
        stars = [" " for _ in range(width)]
        for col in range(width):
            seed = (col * 73 + row * 137 + col * col * 11) % 127
            if seed in (0, 1, 4):
                stars[col] = "✦" if seed == 0 else "·"
        start = max(0, width - 12)
        sky.append(
            style("".join(stars[:start]), "38;2;154;190;219")
            + style(earth[row], "1;38;2;67;151;222")
            + style("".join(stars[start + len(earth[row]) :]), "38;2;154;190;219")
        )

    left: list[tuple[str, str]] = [
        ("Welcome", "1;38;2;210;246;250"),
        ("", "38;2;130;155;171"),
        ("    ╔════╗       ╭──────────────╮", "38;2;255;214;83"),
        ("   ╔╝▓▓▓▓╚╗      │ Ready to     │", "38;2;255;214;83"),
        ("   ║ ▀  ▀ ║      │ explore      │", "38;2;255;214;83"),
        ("   ╚╗ ▿  ╔╝      │ together!    │", "38;2;255;214;83"),
        ("    ╚╦══╦╝       ╰──╮───────────╯", "38;2;255;214;83"),
        ("     ║██║           ╰", "38;2;255;214;83"),
        ("     ║██║", "38;2;255;214;83"),
        ("     ╚══╝", "38;2;255;214;83"),
        ("", "38;2;130;155;171"),
    ]
    if inline_portraits or inline_trainer:
        left = [left[0], *(("", "38;2;130;155;171") for _ in range(13))]
    right: list[tuple[str, str]] = [("▣ PIKA STATUS", "1;38;2;67;222;232")]
    right.extend(
        (f"{label:<9} {value}", "38;2;226;235;242")
        for label, value in (status or (
            ("Model", "loading"), ("Power", "standby"),
            ("Context", "waiting"), ("Mode", "setup"),
        ))
    )
    if compact:
        right.extend([
            ("▣ Tips  /plan  /light  /help", "1;38;2;67;222;232"),
            ("/new  /memories  /trace", "38;2;154;218;231"),
        ])
    else:
        right.extend([
            ("", "38;2;130;155;171"),
            ("▣ Tips for getting started", "1;38;2;67;222;232"),
            ("/plan       enter planning mode", "38;2;154;218;231"),
            ("/light      use execution-state mode", "38;2;154;218;231"),
            ("/help       list commands", "38;2;154;218;231"),
            ("/new        start a fresh conversation", "38;2;154;218;231"),
            ("/memories   inspect long-term memory", "38;2;154;218;231"),
            ("/trace      inspect a run and reflection", "38;2;154;218;231"),
        ])

    if large_titles and three_columns:
        blank_title = ("", "38;2;130;155;171")
        right.insert(1, blank_title)
        tips_row = next(index for index, (text, _) in enumerate(right) if text.startswith("▣ Tips"))
        right.insert(tips_row + 1, blank_title)

    if not three_columns:
        right.append(("", "38;2;130;155;171"))
        right.append(("▣ Choose your mode · 输入 1-4", "1;38;2;67;222;232"))
        if compact:
            right.extend([
                ("1 路卡利欧  2 胖丁", "38;2;154;218;231"),
                ("3 可达鸭    4 皮卡丘", "38;2;154;218;231"),
            ])
        else:
            right.extend(
                (
                    f"{key} {name} · {trait}; {purpose} [{_MODE_BADGES[key][0]}]",
                    "38;2;154;218;231",
                )
                for key, name, trait, purpose in _AGENT_CHOICES
            )
    frame = "1;38;2;67;222;232"

    def edge(text: str) -> str:
        return style(text, frame, background=True)

    lines = (["", *sky] if scene else []) + [edge("╔" + "═" * (width - 2) + "╗")]
    chrome = "▣  ▪ ▪ ▪    Pikachu CLI · local agent workspace"
    lines.append(edge("║") + heading_cell(" " + chrome, width - 2, top=True) + edge("║"))
    if large_titles:
        lines.append(edge("║") + heading_cell("", width - 2, top=True) + edge("║"))
    lines.append(edge("╠" + "═" * (width - 2) + "╣"))
    if split:
        lunar_window = (
            "      ░░▒▒▒▒▒░░▒▒▒░░",
            "  ░▒▒▒◜___◝▒▒░░▒▒▒▒",
            "▒▒▒▒▒▒◟▒▒▒◞▒▒▒▒▒▒▒▒",
        )
        portrait_width, portrait_height = _portrait_size(
            compact=compact or width < 130, inline=inline_portraits
        )
        portrait_lines = portrait_height // 2
        chooser_header_rows = 2 if large_titles else 1
        chooser_rows = chooser_header_rows + portrait_lines * len(_AGENT_CHOICES) if three_columns else 0
        if three_columns:
            blank = ("", "38;2;130;155;171")
            mascot_top = _trainer_art_row(chooser_rows) - 2
            left = [left[0], *([blank] * mascot_top), *left[1:]]
        for index in range(max(len(left), len(right), chooser_rows)):
            if index < len(left):
                left_text, left_color = left[index]
                if three_columns and width < 130 and index >= 2:
                    left_text = left_text.strip().center(left_width)
            else:
                scene_row = index - len(left)
                left_text = (
                    lunar_window[scene_row]
                    if scene and scene_row < len(lunar_window)
                    else ""
                )
                left_color = "38;2;105;112;121"
            right_text, right_color = right[index] if index < len(right) else ("", "38;2;130;155;171")
            left_cell = (
                heading_cell(left_text, left_width)
                if large_titles and index in (0, 1)
                else cell(left_text, left_width, left_color)
            )
            line = edge("║ ") + left_cell
            if three_columns:
                if index == 0:
                    line += edge(" ║ ") + heading_cell("Choose your mode · 1-4", chooser_width)
                elif large_titles and index == 1:
                    line += edge(" ║ ") + heading_cell("", chooser_width)
                elif index >= chooser_rows:
                    line += edge(" ║ ") + cell("", chooser_width, "38;2;185;235;241")
                else:
                    option = (index - chooser_header_rows) // portrait_lines
                    portrait_row = (index - chooser_header_rows) % portrait_lines
                    key, name, trait, purpose = _AGENT_CHOICES[option]
                    label = name if portrait_row == 0 else ""
                    color = "1;38;2;255;214;83" if key == selected_agent else "38;2;185;235;241"
                    marker = f"{'›' if key == selected_agent else ' '}{key} " if portrait_row == 0 else "   "
                    if inline_portraits:
                        portrait = cell("", portrait_width, color)
                    else:
                        portrait = _portrait_line(
                            key, portrait_row, use_color=use_color,
                            width=portrait_width, height=portrait_height,
                        )
                    label_width = chooser_width - 3 - portrait_width
                    if portrait_row == 2:
                        badge_name, badge_bg, badge_fg = _MODE_BADGES[key]
                        badge_text = f" {badge_name.center(8)} "
                        description_width = label_width - len(badge_text)
                        description_cell = cell(" " + purpose, description_width, color)
                        if use_color:
                            foreground = ";".join(map(str, badge_fg))
                            background = ";".join(map(str, badge_bg))
                            label_cell = (
                                description_cell
                                + f"\033[1;38;2;{foreground};48;2;{background}m{badge_text}\033[0m"
                            )
                        else:
                            label_cell = description_cell + f"[{badge_name.center(8)}]"
                    elif portrait_row == 1:
                        label_cell = cell(" " + trait, label_width, color)
                    else:
                        label_cell = cell(" " + label, label_width, color)
                    line += edge(" ║ ") + cell(marker, 3, color) + portrait + label_cell
            status_cell = (
                heading_cell(right_text, right_width)
                if right_text.startswith("▣ ")
                else cell(right_text, right_width, right_color)
            )
            line += edge(" ║ ") + status_cell
            lines.append(line + edge(" ║"))
    else:
        for text, color in [*left, ("", "38;2;130;155;171"), *right]:
            body_cell = heading_cell(text, left_width) if text.startswith("▣ ") else cell(text, left_width, color)
            lines.append(edge("║ ") + body_cell + edge(" ║"))
    lines.append(edge("╠" + "═" * (width - 2) + "╣"))
    lines.append(edge("║ ") + cell("› Type a message to begin.", width - 4, "3;38;2;184;203;212") + edge(" ║"))
    lines.append(edge("╚" + "═" * (width - 2) + "╝"))

    if scene:
        ground_rows = 1 if terminal_lines <= 26 else 2
        lunar = ["░" if (col * 19 + row * 41) % 11 < 3 else "▒" for row in range(ground_rows) for col in range(width)]
        for row in range(ground_rows):
            surface = lunar[row * width : (row + 1) * width]
            for position in (width // 6, width // 2, width * 5 // 6):
                crater = "◜___◝" if row == 0 else "◟▒▒▒◞"
                position = min(position, width - len(crater))
                surface[position : position + len(crater)] = list(crater)
            lines.append(style("".join(surface), "38;2;126;128;133"))
        lines.append("")
    return "\n".join(lines)


def format_power_bar(used_tokens: int, capacity_tokens: int, *, width: int = 10) -> str:
    """Render remaining context capacity as a compact Pikachu power bar."""
    capacity = max(1, capacity_tokens)
    used = max(0, min(used_tokens, capacity))
    remaining_percent = round((capacity - used) / capacity * 100)
    filled = round(remaining_percent / 100 * width)
    return f"{'█' * filled}{'░' * (width - filled)} {remaining_percent}%"


def print_live_usage(
    *, context_used: int, context_window: int, run_tokens: int, session_tokens: int
) -> None:
    """Show measured model usage and the latest estimated context size."""
    max_width = max(20, terminal_size((100, 24)).columns - 2)
    lines = (
        f"Context ≈{context_used:,}/{context_window:,} · Power {format_power_bar(context_used, context_window)}",
        f"Tokens · 本轮 {run_tokens:,} · 本次 CLI 会话累计 {session_tokens:,}",
    )
    width = min(max_width, max(44, *(_terminal_display_width(line) + 4 for line in lines)))
    print(_ui_style("\n╭─ PIKA STATUS " + "─" * max(0, width - 16) + "╮", "38;5;117"))
    for line in lines:
        fitted = _fit_terminal_text(line, width - 4)
        print(
            _ui_style("│", "38;5;117") + " "
            + _ui_style(fitted, "38;5;252")
            + " " * (width - 4 - _terminal_display_width(fitted))
            + " " + _ui_style("│", "38;5;117")
        )
    print(_ui_style("╰" + "─" * (width - 2) + "╯", "38;5;117"))


def print_startup_status(
    rows: Sequence[tuple[str, str]],
    *,
    notices: Sequence[str] = (),
    compact: bool = False,
) -> None:
    use_color = _terminal_ui_color_enabled()
    accent_color = "38;5;117"

    def style(text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if use_color else text

    if compact:
        width = max(20, terminal_size((80, 24)).columns - 2)
        for start in range(0, len(rows), 2):
            line = " · ".join(f"{label}：{value}" for label, value in rows[start : start + 2])
            print(style(_fit_terminal_text(line, width), "38;5;252"))
        for notice in notices:
            print(f"{style('提醒：', '38;5;214')}{notice}")
        print(style("输入任务开始工作 · /help 查看命令 · /exit 退出", "38;5;245"))
        return

    print()
    for label, value in rows:
        label_text = style(f"  • {label}：", f"1;{accent_color}")
        print(f"{label_text}{style(value, '38;5;252')}")
    for notice in notices:
        print(f"{style('  提醒：', '38;5;214')}{notice}")
    print(
        f"\n{style('  输入任务开始工作 · /plan 规划模式 · /help 查看命令 · /new 新建会话 · /exit 退出', '38;5;245')}"
    )


def print_conversation_divider(label: str = "会话") -> None:
    """Print a visible boundary between conversations or input turns."""
    columns = terminal_size((80, 24)).columns
    width = max(48, min(columns, 120))
    marker = f" {label} "
    remaining = max(0, width - len(marker) - 2)
    left = remaining // 2
    right = remaining - left
    print(f"\n{'─' * left}{marker}{'─' * right}")


async def run_setup(
    *,
    settings: ModelConfig | None = None,
    config_path: str | Path | None = None,
    input_fn: Callable[[str], str] = input,
    secret_fn: Callable[[str], str] = getpass.getpass,
    output_fn: Callable[[str], Any] = print,
) -> bool:
    path = Path(config_path or ModelConfig.model_config["env_file"]).expanduser().absolute()
    try:
        current = settings if settings is not None else ModelConfig(_env_file=path)
        print_banner(output_fn=output_fn)
        output_fn(f"配置保存到：{path}")
        output_fn("API Key 将保存在该配置文件中，文件权限设为仅当前用户读写。")
        output_fn("进程环境变量优先于文件；已有环境变量覆盖时，需同步调整环境变量。")
        selected = await _choose_provider(current, input_fn, output_fn)
        prefix = selected.value
        old_model = getattr(current, f"{prefix}_model")
        old_url = getattr(current, f"{prefix}_base_url")
        old_key = getattr(current, f"{prefix}_api_key")

        model = (await asyncio.to_thread(input_fn, f"模型名称 [{old_model}]：")).strip() or old_model
        url_input = (await asyncio.to_thread(
            input_fn, f"API 地址 [{old_url or '默认地址'}]（回车保留）："
        )).strip()
        base_url = url_input or old_url
        has_key = old_key is not None and bool(old_key.get_secret_value().strip())
        hint = "回车保留现有密钥" if has_key else "必填，输入不回显"
        new_key = (await asyncio.to_thread(secret_fn, f"API Key（{hint}）：")).strip()
        if not new_key and not has_key:
            output_fn("未提供 API Key，设置已取消。")
            return False

        # 用新值构造配置，不修改调用方传入的 settings 对象。
        values = current.model_dump()
        values.update({
            "model_default_provider": selected,
            f"{prefix}_model": model,
            f"{prefix}_base_url": base_url,
        })
        if new_key:
            values[f"{prefix}_api_key"] = SecretStr(new_key)
        candidate = ModelConfig.model_validate(values)
        candidate.load_provider_config(selected)

        output_fn(f"\n将保存：{_PROVIDER_LABELS[selected]} / {model}")
        output_fn(f"API 地址：{base_url or '默认地址'}")
        if not await _confirm("确认保存？[Y/n] ", input_fn):
            output_fn("设置已取消。")
            return False
        updates = {
            "MODEL_DEFAULT_PROVIDER": selected.value,
            f"{prefix.upper()}_MODEL": model,
            f"{prefix.upper()}_BASE_URL": base_url or "",
        }
        # 回车保留密钥时，不把环境变量中的密钥额外复制到文件中。
        if new_key:
            updates[f"{prefix.upper()}_API_KEY"] = new_key
        try:
            await asyncio.to_thread(_save_config, path, updates)
        except Exception as exc:
            output_fn(f"保存失败（{type(exc).__name__}），原配置未被替换。")
            return False
        # 配置向导可能是在当前 Pikachu 进程里触发的。下一步会立即重新
        # 创建 Application；把本次明确选择的 provider 同步到当前进程，
        # 避免旧的 MODEL_DEFAULT_PROVIDER 环境变量继续把选择覆盖回去。
        os.environ["MODEL_DEFAULT_PROVIDER"] = selected.value
        output_fn("配置已保存。")

        if await _confirm("立即测试连接？[Y/n] ", input_fn):
            output_fn("正在测试模型连接……")
            try:
                elapsed = await _test_connection(candidate, selected)
            except Exception as exc:
                # Provider SDK 的异常正文不包含 API Key，保留它才能区分
                # 模型不存在、认证失败、请求路径错误和网络错误。
                output_fn(
                    f"连接测试失败（{type(exc).__name__}）：{exc}\n"
                    "已保存的配置保留。"
                )
            else:
                output_fn(f"连接成功：{selected.value}/{model} · {elapsed:.0f}ms")
        return await _confirm("现在进入 Pikachu？[Y/n] ", input_fn)
    except (EOFError, KeyboardInterrupt):
        output_fn("\n设置已取消。")
        return False


async def _choose_provider(
    settings: ModelConfig,
    input_fn: Callable[[str], str],
    output_fn: Callable[[str], Any],
) -> ModelProvider:
    providers = tuple(_PROVIDER_LABELS)
    default = settings.model_default_provider
    if default not in providers:
        default = providers[0]
    default_index = providers.index(default) + 1
    output_fn("选择主模型提供方：")
    for index, provider in enumerate(providers, 1):
        key = getattr(settings, f"{provider.value}_api_key")
        configured = key is not None and bool(key.get_secret_value().strip())
        marker = "（已配置）" if configured else ""
        output_fn(f"  {index}. {_PROVIDER_LABELS[provider]} {marker}")
    while True:
        value = (await asyncio.to_thread(input_fn, f"请选择 [{default_index}]：")).strip().lower()
        if not value:
            return default
        if value.isdigit() and 1 <= int(value) <= len(providers):
            return providers[int(value) - 1]
        for provider in providers:
            if value == provider.value:
                return provider
        output_fn("请输入列表编号或提供方名称。")


async def _confirm(prompt: str, input_fn: Callable[[str], str]) -> bool:
    while True:
        answer = (await asyncio.to_thread(input_fn, prompt)).strip().lower()
        if answer in {"", "y", "yes", "是"}:
            return True
        if answer in {"n", "no", "否"}:
            return False
        prompt = "请输入 y 或 n："


def _save_config(path: Path, updates: dict[str, str]) -> None:
    """在临时文件更新选中字段，再原子替换；保留其他设置和注释。"""
    if path.is_symlink():
        raise ValueError("config file cannot be a symbolic link")
    path.parent.mkdir(parents=True, exist_ok=True)
    content = path.read_text(encoding="utf-8") if path.exists() else ""
    descriptor, filename = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(filename)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
        for key, value in updates.items():
            set_key(str(temporary), key, value, quote_mode="always", encoding="utf-8")
        temporary.chmod(0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_search_api_key(api_key: str) -> None:
    """保存交互式搜索配置，不回显或记录 key。"""

    path = Path(SearchSettings.model_config["env_file"]).expanduser().absolute()
    _save_config(path, {"TAVILY_API_KEY": api_key})


async def _test_connection(settings: ModelConfig, provider: ModelProvider) -> float:
    # 仅选择测试时加载模型依赖，配置界面不依赖 Registry 或设置服务。
    from app.model.config import Message, MessageRole
    from app.model.adapter import ModelCompatibleAdapter
    from app.model.config import ModelRequest

    adapter = ModelCompatibleAdapter(settings.load_provider_config(provider))
    started = perf_counter()
    try:
        await adapter.complete(ModelRequest(
            messages=(Message(role=MessageRole.USER, content="请只回复 OK。"),),
            model=adapter.default_model,
            max_output_tokens=64,
        ))
        return (perf_counter() - started) * 1000
    finally:
        await adapter.close()


def _terminal_ui_color_enabled() -> bool:
    """Keep Pikachu's iTerm UI colored even when NO_COLOR is inherited."""
    if not getattr(sys.stdout, "isatty", lambda: False)():
        return False
    if not os.environ.get("NO_COLOR"):
        return True
    return (
        os.environ.get("TERM_PROGRAM", "").lower() == "iterm.app"
        or os.environ.get("LC_TERMINAL") == "iTerm2"
    )


def _ui_style(text: str, code: str) -> str:
    """Apply terminal color when Pikachu's UI supports it."""
    if not _terminal_ui_color_enabled():
        return text
    return f"\033[{code}m{text}\033[0m"


def _compact_ui_value(value: Any, *, limit: int = 260) -> str:
    """Keep tool arguments/results readable without flooding the terminal."""
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        try:
            text = json.dumps(value, ensure_ascii=False, separators=(", ", ": "))
        except (TypeError, ValueError):
            text = str(value)
    else:
        text = str(value)
    text = " ".join(text.split())
    if len(text) > limit:
        return text[: limit - 1] + "…"
    return text


def _print_event_block(
    title: str,
    lines: Sequence[str] = (),
    *,
    color: str = "38;5;245",
    title_color: str = "38;5;252",
) -> None:
    """Render one loop event as a Claude-style vertical block."""
    bar = _ui_style("│", color)
    print()
    print(f"{bar} {_ui_style(title, title_color)}")
    for line in lines:
        for physical_line in str(line).splitlines() or [""]:
            print(f"{bar}   {physical_line}")


def print_agent_event(event: AgentEvent) -> None:
    """把 Runtime 事件渲染成按 loop 步骤分隔的终端事件块。"""

    step = f" · step {event.step}" if event.step is not None else ""
    if event.type == "agent_started":
        _print_event_block("assistant · started", ["开始执行当前 Run"], color="38;5;114")
    elif event.type == "model_started":
        lines = ["正在请求模型"]
        if event.compaction_stage not in (None, "none"):
            lines.append(f"上下文处理：{event.compaction_stage}")
        if event.prepared_input_tokens is not None:
            lines.append(f"输入≈{event.prepared_input_tokens} tokens")
        _print_event_block(
            f"model · running{step}", lines, color="38;5;117", title_color="38;5;117"
        )
    elif event.type == "model_completed":
        tool_count = len(event.message.tool_calls) if event.message else 0
        usage = event.usage
        usage_line = (
            f"本次调用：输入 {usage.input_tokens:,} · 输出 {usage.output_tokens:,} · 合计 {usage.total_tokens:,} tokens"
            if usage is not None else None
        )
        if tool_count:
            names = ", ".join(call.name for call in event.message.tool_calls)
            _print_event_block(
                f"model · tool calls{step}",
                [f"请求 {tool_count} 个工具：{names}", *([usage_line] if usage_line else [])],
                color="38;5;117", title_color="38;5;117",
            )
        else:
            _print_event_block(
                "model · completed", ["模型已返回回复", *([usage_line] if usage_line else [])],
                color="38;5;117", title_color="38;5;117",
            )
    elif event.type == "run_budget_warning":
        _print_event_block(
            "budget · warning",
            [f"{event.run_budget_chargeable_tokens or 0} tokens · {event.run_budget_model_calls or 0} calls"],
            color="38;5;214", title_color="38;5;214",
        )
    elif event.type == "run_budget_finalizing":
        _print_event_block("budget · finalizing", ["达到收口线，正在生成最终答复"], color="38;5;214", title_color="38;5;214")
    elif event.type == "run_budget_exceeded":
        _print_event_block("budget · exceeded", ["达到硬上限，停止继续请求模型"], color="38;5;203", title_color="38;5;203")
    elif event.type == "tool_started" and event.tool_call:
        arguments = _compact_ui_value(event.tool_call.arguments)
        _print_event_block(
            f"tool {event.tool_call.name} · running{step}",
            [f"参数：{arguments}" if arguments else "正在调用工具"],
        )
    elif event.type == "tool_completed" and event.tool_result:
        status = "成功" if event.tool_result.success else "失败"
        result_lines = [f"耗时：{event.tool_result.duration_ms:.1f}ms"]
        output = _compact_ui_value(
            event.tool_result.output if event.tool_result.success else event.tool_result.error
        )
        if output:
            result_lines.append(f"结果：{output}")
        result_color = "38;5;114" if event.tool_result.success else "38;5;203"
        _print_event_block(
            f"tool {event.tool_result.tool_name} · {status}",
            result_lines, color=result_color, title_color=result_color,
        )
    elif event.type == "tool_approval_required" and event.tool_call:
        _print_event_block(
            f"permission · requested · {event.tool_call.name}",
            ["等待人工审批"], color="38;5;214", title_color="38;5;214",
        )
    elif event.type == "tool_approval_completed" and event.tool_call:
        decision = event.approval_decision.value if event.approval_decision else "unknown"
        rule = event.rule_description or "权限检查完成"
        decision_color = "38;5;114" if decision in {"allow", "allowed"} else "38;5;214"
        _print_event_block(
            f"permission · {decision} · {event.tool_call.name}",
            [f"规则：{rule}"], color=decision_color, title_color=decision_color,
        )
    elif event.type == "memory_reflection_started":
        _print_event_block("memory · reflection", ["正在整理本轮长期记忆"], color="38;5;183", title_color="38;5;183")
    elif event.type == "memory_reflection_completed":
        action = event.reflection_action or "none"
        suffix = f" · {event.reflection_memory_id}" if event.reflection_memory_id else ""
        _print_event_block("memory · reflection completed", [f"动作：{action}{suffix}"], color="38;5;183", title_color="38;5;183")
    elif event.type == "memory_reflection_failed":
        _print_event_block("memory · reflection failed", ["整理失败，已跳过"], color="38;5;203", title_color="38;5;203")
    elif event.type == "memory_reflection_skipped":
        _print_event_block("memory · reflection skipped", [event.reflection_skip_reason or "policy"])
    elif event.type == "memory_archive_started":
        _print_event_block("memory · archive", ["长期记忆容量不足，正在选择可归档候选"], color="38;5;221", title_color="38;5;221")
    elif event.type == "memory_archive_completed":
        action = event.archive_action or "unknown"
        suffix = f" · {event.archive_memory_id}" if event.archive_memory_id else ""
        _print_event_block("memory · archive completed", [f"动作：{action}{suffix}"], color="38;5;221", title_color="38;5;221")
    elif event.type == "memory_archive_failed":
        _print_event_block("memory · archive failed", ["容量维护失败，未执行归档"], color="38;5;203", title_color="38;5;203")
    elif event.type == "memory_archive_skipped":
        _print_event_block("memory · archive skipped", [event.archive_skip_reason or "policy"])
    elif event.type == "skill_activated":
        _print_event_block("skill · activated", [event.skill_name or "unknown"], color="38;5;117", title_color="38;5;117")
    elif event.type == "skill_activation_failed":
        _print_event_block("skill · activation failed", [f"{event.skill_name or 'unknown'} · {event.skill_error or '未知原因'}"], color="38;5;203", title_color="38;5;203")
    elif event.type == "context_handoff":
        _print_event_block("context · handoff", ["正在切换上下文并保留运行状态"], color="38;5;183", title_color="38;5;183")
    elif event.type == "context_compacted":
        color = "38;5;203" if event.summary_error else "38;5;183"
        if event.summary_error:
            lines = [f"压缩失败：{event.summary_error}"]
        else:
            lines = [
                f"压缩阶段：{event.compaction_stage or 'unknown'}",
                f"摘要覆盖：{event.summarized_conversation_blocks or 0} 个对话块",
                f"截短工具结果：{event.compacted_tool_results or 0} 条",
                f"移除旧工具轮次：{event.removed_tool_rounds or 0} 个",
            ]
            if event.summary_updated and event.summary_text:
                lines.extend(["", "摘要：", *event.summary_text.splitlines()])
            elif not any(
                (
                    event.summary_updated,
                    event.compacted_tool_results,
                    event.removed_tool_rounds,
                )
            ):
                lines.append("当前没有可压缩的旧对话，或摘要没有发生变化。")
        _print_event_block(
            "context · compacted",
            lines,
            color=color,
            title_color=color,
        )
    elif event.type == "plan_updated":
        lines = []
        for item in event.plan_steps:
            status = str(item.get("status") or "todo")
            marker = {
                "done": "✓",
                "in_progress": "→",
                "blocked": "!",
            }.get(status, " ")
            lines.append(f"[{marker}] {item.get('title', item.get('id', ''))}")
        _print_event_block(
            "plan · todo",
            lines or ["暂无步骤"],
            color="38;5;117",
            title_color="38;5;117",
        )
    elif event.type == "agent_completed":
        _print_event_block("assistant · completed", ["当前 Run 执行完成"], color="38;5;114", title_color="38;5;114")
    elif event.type == "agent_failed":
        reason = event.stop_reason.value if event.stop_reason else "unknown"
        _print_event_block("assistant · failed", [f"执行停止：{reason}"], color="38;5;203", title_color="38;5;203")
    elif event.type == "agent_cancelled":
        _print_event_block("assistant · cancelled", ["当前 Run 已取消"], color="38;5;214", title_color="38;5;214")


def print_assistant_message(content: str) -> None:
    """Render the final assistant response with the same vertical guide."""
    _print_event_block(
        "assistant · final answer",
        content.splitlines() or [""],
        color="38;5;114",
        title_color="38;5;114",
    )

def print_checkpoints(checkpoints: tuple[RunCheckpoint, ...]) -> None:
    """显示当前会话最近的恢复边界。"""

    if not checkpoints:
        print("当前会话暂无 Checkpoint。")
        return
    for checkpoint in checkpoints:
        updated_at = checkpoint.updated_at.astimezone().strftime("%m-%d %H:%M:%S")
        pending = len(checkpoint.pending_tool_calls)
        print(
            f"{checkpoint.run_id[:8]}  {updated_at}  "
            f"{checkpoint.status.value:<11} phase={checkpoint.phase.value} "
            f"step={checkpoint.step} pending_tools={pending}"
        )

def print_recovered_runs(runs: Sequence[Run]) -> None:
    """展示启动时修正的 Run 状态；不会在此恢复执行。"""
    for run in runs:
        if run.status == "interrupted":
            print(f"检测到中断 Run：{run.id[:8]} · 已保留恢复检查点")
        else:
            print(
                f"Run 状态修正：{run.id[:8]} → {run.status.value}"
                + (f" · {run.error}" if run.error else "")
            )

def print_trace(events: tuple[AgentEvent, ...]) -> None:
    """显示一次 Run 的完整事件时间线。"""

    for event in events:
        event_time = event.event_time.astimezone().strftime("%H:%M:%S.%f")[:-3]
        details: list[str] = []
        if event.step is not None:
            details.append(f"step={event.step}")
        if event.tool_call is not None:
            details.append(f"tool={event.tool_call.name}")
        if event.approval_decision is not None:
            details.append(f"decision={event.approval_decision.value}")
        if event.rule_id is not None:
            details.append(f"rule={event.rule_id[:8]}")
        if event.tool_result is not None:
            details.append(
                f"success={'true' if event.tool_result.success else 'false'}"
            )
        if event.type == 'model_started':
            if event.prepared_input_tokens is not None:
                details.append(f"input≈{event.prepared_input_tokens}")
            if event.tool_schema_tokens is not None:
                details.append(f"schemas≈{event.tool_schema_tokens}")
            if (
                event.tool_result_tokens_before is not None
                and event.tool_result_tokens_after is not None
            ):
                details.append(
                    "tool_results≈"
                    f"{event.tool_result_tokens_before}→"
                    f"{event.tool_result_tokens_after}"
                )
            if event.compaction_stage not in (None, "none"):
                details.append(f"context={event.compaction_stage}")
            if event.summary_provider is not None:
                details.append(
                    "summary="
                    f"{event.summary_provider}/{event.summary_model or 'default'}"
                )
            if event.summary_duration_ms is not None:
                details.append(f"summary_ms={event.summary_duration_ms:.1f}")
        if event.run_budget_status is not None:
            details.append(
                f"budget={event.run_budget_status}:"
                f"{event.run_budget_chargeable_tokens or 0}t/"
                f"{event.run_budget_model_calls or 0}calls"
            )
        detail_text = f"  {' '.join(details)}" if details else ""
        print(f"{event.sequence:03d}  {event_time}  {event.type.value}{detail_text}")

def print_permission_rules(rules: tuple[PermissionRule, ...]) -> None:
    """显示当前会话记住的工具审批规则。"""

    if not rules:
        print("当前会话没有已记住的审批规则。")
        return
    for rule in rules:
        created_at = rule.created_at.astimezone().strftime("%m-%d %H:%M:%S")
        print(f"{rule.id[:8]}  {created_at}  {rule.tool_name}  {rule.description}")

def print_memories(memories: Sequence[RegularMemoryRecord]) -> None:
    """输出适合终端快速浏览的长期记忆列表（Recall Cue）。"""

    if not memories:
        print("没有长期记忆。")
        return
    for memory in memories:
        cue = " ".join(memory.summary.split())
        print(f"{memory.id}  [{memory.status.value}]  {memory.title}")
        print(f"  Cue: {cue}")

def print_memory(memory: RegularMemoryRecord) -> None:
    """输出单条长期记忆的完整内容。"""

    print(
        f"ID: {memory.id}\n"
        f"状态: {memory.status.value}\n"
        f"标题: {memory.title}\n"
        f"访问次数: {memory.access_count}\n"
        f"创建时间: {memory.created_at.astimezone().isoformat()}\n"
        f"内容:\n{memory.render_full()}"
    )

def print_help() -> None:
    """仅列出 chat.py 当前已实现的交互命令。"""
    print(
        "可用命令：\n"
        "  /new [标题]       新建会话\n"
        "  /memories         查看记忆列表\n"
        "  /memory <记忆ID>  查看完整记忆\n"
        "  /permissions      查看当前会话的审批规则\n"
        "  /checkpoints      查看当前会话的检查点\n"
        "  /trace <Run ID>   查看运行事件\n"
        "  /clear            清空当前会话消息和摘要\n"
        "  /compact          手动压缩上下文，保留近期对话\n"
        "  /plan             进入 PLAN MODE（只规划，不修改文件）\n"
        "  /plan off         退出 PLAN MODE，恢复普通执行\n"
        "  /light            进入 LIGHT MODE，使用结构化执行状态\n"
        "  /light off        退出 LIGHT MODE，恢复普通执行\n"
        "  /help             显示帮助\n"
        "  /exit 或 /quit    退出聊天"
    )


__all__ = ['print_banner', 'format_power_bar', 'print_live_usage', 'print_startup_status', 'print_conversation_divider', 'run_setup', 'save_search_api_key', 'print_agent_event', 'print_assistant_message', 'print_recovered_runs', 'print_memories', 'print_memory', 'print_permission_rules', 'print_checkpoints', 'print_trace', 'print_help']

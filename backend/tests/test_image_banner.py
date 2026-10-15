"""The image banner must leave normal CLI output usable on every terminal."""

from __future__ import annotations

import asyncio
import base64
import io
import os
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image
from app.agent.events import AgentEvent, AgentEventType
from app.interfaces.cli.chat import _run_interactive, _send_message
from app.interfaces.cli.cli_ui import (
    _ui_style,
    _portrait_line,
    _portrait_pixels,
    _render_banner_text,
    _terminal_display_width,
    print_agent_event,
    print_banner,
    print_live_usage,
    print_startup_status,
    terminal_size,
)
from app.interfaces.cli.image_banner import (
    BACKGROUND,
    PANEL_BACKGROUND,
    _background_sequence,
    _inline_image_sequence,
    _portrait_overlays,
    _title_image_sequence,
    _title_overlays,
    _trainer_overlays,
    clear_image_banner,
    image_protocol,
)
from app.model.config import ModelUsage
from app.model.config import AgentMode


STATUS = (
    ("Model", "deepseek/deepseek-flash"),
    ("Power", "█████████░ 99%"),
    ("Context", "5,581 / 1,048,576"),
    ("Mode", "default"),
)


def test_protocol_selection() -> None:
    assert image_protocol({"TERM_PROGRAM": "iTerm.app"}) == "iterm"
    assert image_protocol({"TERM": "xterm-kitty"}) is None
    assert image_protocol({"TERM": "xterm-256color"}) is None
    assert image_protocol({"TERM_PROGRAM": "iTerm.app", "TMUX": "1"}) is None
    assert image_protocol({"TERM_PROGRAM": "iTerm.app", "NO_COLOR": "1"}) == "iterm"
    assert image_protocol({"TERM_PROGRAM": "iTerm.app", "PIKACHU_BANNER": "ascii"}) is None


def test_iterm_ui_keeps_event_colors_when_no_color_is_inherited() -> None:
    class TerminalOutput(io.StringIO):
        def isatty(self) -> bool:
            return True

    with (
        patch.dict(os.environ, {"TERM_PROGRAM": "iTerm.app", "NO_COLOR": "1"}, clear=True),
        patch("sys.stdout", TerminalOutput()),
    ):
        assert _ui_style("model · running", "38;5;117") == (
            "\x1b[38;5;117mmodel · running\x1b[0m"
        )

    with (
        patch.dict(os.environ, {"TERM_PROGRAM": "other", "NO_COLOR": "1"}, clear=True),
        patch("sys.stdout", TerminalOutput()),
    ):
        assert _ui_style("model · running", "38;5;117") == "model · running"


def test_iterm_background_uses_path_not_inline_image() -> None:
    sequence = _background_sequence(BACKGROUND)
    assert sequence.startswith("\x1b]1337;SetBackgroundImageFile=")
    encoded = sequence.split("=", 1)[1][:-1]
    assert os.fsdecode(base64.b64decode(encoded)) == str(BACKGROUND.resolve())
    assert _background_sequence(None) == "\x1b]1337;SetBackgroundImageFile=\x07"


def test_iterm_portrait_sends_original_png_in_reserved_cells() -> None:
    portrait = BACKGROUND.with_name("agent_jigglypuff.png")
    with Image.open(portrait) as image:
        assert image.size == (256, 256)
        assert image.mode == "RGBA"
    sequence = _inline_image_sequence(portrait, width=8, height=4)
    header, encoded = sequence[:-1].split(":", 1)
    assert header.startswith("\x1b]1337;File=inline=1;")
    assert "width=8;height=4;preserveAspectRatio=1" in header
    assert base64.b64decode(encoded) == portrait.read_bytes()


def test_trainer_transparency_uses_the_welcome_panel_color() -> None:
    trainer = BACKGROUND.with_name("trainer_ash.png")
    sequence = _inline_image_sequence(
        trainer, width=14, height=9, background=PANEL_BACKGROUND,
    )
    assert "width=14;height=9;preserveAspectRatio=0" in sequence
    encoded = sequence[:-1].split(":", 1)[1]
    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        assert image.mode == "RGB"
        assert image.getpixel((0, 0)) == PANEL_BACKGROUND
        assert image.size == (112, 144)


def test_portrait_and_large_title_cover_their_cells_with_panel_color() -> None:
    portrait = BACKGROUND.with_name("agent_jigglypuff.png")
    for sequence, size in (
        (_inline_image_sequence(
            portrait, width=8, height=4, background=PANEL_BACKGROUND,
        ), (64, 64)),
        (_title_image_sequence("▣ PIKA STATUS", 40), (320, 32)),
    ):
        assert "preserveAspectRatio=0" in sequence
        encoded = sequence[:-1].split(":", 1)[1]
        with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
            assert image.mode == "RGB"
            assert image.size == size
            assert image.getpixel((0, 0)) == PANEL_BACKGROUND
            if size == (320, 32):
                assert image.getpixel((image.width - 1, image.height - 1)) == PANEL_BACKGROUND


def test_text_banner_remains_available_without_image_output() -> None:
    output: list[str] = []
    with patch("shutil.get_terminal_size", return_value=os.terminal_size((100, 24))):
        print_banner(output_fn=output.append, status=STATUS)
    assert "Welcome" in output[0]
    assert "PIKA STATUS" in output[0]
    assert "/memories" in output[0]
    assert "路卡利欧" in output[0] and "皮卡丘" in output[0]


def test_tty_welcome_keeps_status_as_terminal_text() -> None:
    class TerminalOutput(io.StringIO):
        def isatty(self) -> bool:
            return True

    output = TerminalOutput()
    with (
        patch.dict(os.environ, {"TERM_PROGRAM": "iTerm.app", "NO_COLOR": "1"}, clear=True),
        patch("sys.stdout", output),
        patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))),
        patch("app.interfaces.cli.image_banner._background_active", False),
    ):
        print_banner(status=STATUS)
        clear_image_banner()
    text = output.getvalue()
    assert text.startswith("\x1b]1337;SetBackgroundImageFile=")
    assert text.count("File=inline=1") == 11
    assert "Welcome" in text
    assert "5,581 / 1,048,576" in text
    assert "\x1b[1;38;2;67;222;232;48;2;21;29;40m" in text
    assert "\x1b[1;38;2;234;246;255;48;2;24;91;156m  NORMAL  " in text
    assert text.endswith(_background_sequence(None))


def test_iterm_default_80_by_24_window_uses_image_and_live_text() -> None:
    class TerminalOutput(io.StringIO):
        def isatty(self) -> bool:
            return True

    output = TerminalOutput()
    with (
        patch.dict(os.environ, {"TERM_PROGRAM": "iTerm.app", "NO_COLOR": "1"}, clear=True),
        patch("sys.stdout", output),
        patch("shutil.get_terminal_size", return_value=os.terminal_size((80, 24))),
        patch("app.interfaces.cli.image_banner._background_active", False),
    ):
        shown = print_banner(status=(*STATUS, ("Tokens", "本次 CLI 会话 168")))
        print_startup_status((
            ("主模型", "deepseek/deepseek-flash"),
            ("会话", "恢复会话 · 3 条消息"),
            ("搜索", "Tavily"),
            ("长期记忆", "Sparse Memory"),
        ), compact=True)
        clear_image_banner()
    text = output.getvalue()
    assert shown is True
    assert "SetBackgroundImageFile=" in text
    assert text.count("File=inline=1") == 2  # Ash and bubble remain visible in two columns
    assert "Ready to" not in text  # the ASCII stand-in is not printed beneath them
    assert "本次 CLI 会话 168" in text
    assert "Tips  /plan  /light  /help" in text
    assert "/new  /memories  /trace" in text
    assert "搜索：Tavily" in text


def test_tall_two_column_iterm_window_keeps_trainer_without_large_title_overlays() -> None:
    class TerminalOutput(io.StringIO):
        def isatty(self) -> bool:
            return True

    output = TerminalOutput()
    with (
        patch.dict(os.environ, {"TERM_PROGRAM": "iTerm.app"}, clear=True),
        patch("sys.stdout", output),
        patch("shutil.get_terminal_size", return_value=os.terminal_size((84, 34))),
        patch("app.interfaces.cli.image_banner._background_active", False),
    ):
        assert print_banner(status=STATUS)
        clear_image_banner()
    text = output.getvalue()
    assert text.count("File=inline=1") == 2
    assert "Welcome" in text
    assert "▣ Choose your mode · 输入 1-4" in text


def test_mid_size_iterm_window_keeps_the_mode_portraits() -> None:
    class TerminalOutput(io.StringIO):
        def isatty(self) -> bool:
            return True

    output = TerminalOutput()
    with (
        patch.dict(os.environ, {"TERM_PROGRAM": "iTerm.app"}, clear=True),
        patch("sys.stdout", output),
        patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 30))),
        patch("app.interfaces.cli.image_banner._background_active", False),
    ):
        assert print_banner(status=STATUS)
        clear_image_banner()
    text = output.getvalue()
    assert "Choose your mode · 1-4" in text
    assert text.count("File=inline=1") == 6
    assert all(name in text for name in ("Lucario", "Jigglypuff", "Psyduck", "Pikachu"))


def test_compact_three_column_banner_centers_mascot_and_status() -> None:
    frame = _render_banner_text(
        (*STATUS, ("Tokens", "本次 CLI 会话 0")),
        width=116, terminal_lines=30, use_color=False,
        scene=False, compact=True, inline_portraits=True,
    )
    rows = frame.splitlines()
    assert "Welcome" in rows[3]
    assert "Choose your mode · 1-4" in rows[3]
    assert "PIKA STATUS" in rows[3]
    assert rows[7].split("║")[1].strip() == ""  # reserved for the two trainer images
    assert "/new  /memories  /trace" in frame
    assert "Pikachu 皮卡丘" in rows[13]


def test_wide_iterm_window_centers_text_and_sets_background_once() -> None:
    class TerminalOutput(io.StringIO):
        def isatty(self) -> bool:
            return True

    output = TerminalOutput()
    with (
        patch.dict(os.environ, {"TERM_PROGRAM": "iTerm.app", "NO_COLOR": "1"}, clear=True),
        patch("sys.stdout", output),
        patch("shutil.get_terminal_size", return_value=os.terminal_size((202, 47))),
        patch("app.interfaces.cli.image_banner._background_active", False),
    ):
        print_banner(status=STATUS)
        print_banner(status=(*STATUS, ("Tokens", "本次 CLI 会话 168")))
        clear_image_banner()
    text = output.getvalue()
    assert text.count("SetBackgroundImageFile=") == 2  # set, then restore
    assert text.count("File=inline=1") == 22  # six sprites and five enlarged headings per redraw
    assert text.count("Welcome") >= 2
    assert "本次 CLI 会话 168" in text
    assert "\n" + " " * 14 + "\x1b[1;38;2;67;222;232;48;2;21;29;40m╔" in text
    assert "Choose your mode · 1-4" in text
    assert all(name in text for name in ("Lucario", "Jigglypuff", "Psyduck", "Pikachu"))
    assert all(description in text for description in (
        "Aura-guided precision", "full-power coding",
        "A voice full of charm", "full-power creation",
        "Ponders every puzzle", "plan before acting",
        "Lightning-fast energy", "quick, light tasks",
    ))
    assert text.count(" NORMAL ") == 4
    assert text.count(" PLAN ") == 2
    assert text.count(" LIGHT ") == 2


def test_portrait_overlays_return_cursor_to_the_live_input_position() -> None:
    frame = _render_banner_text(
        STATUS, width=174, terminal_lines=47, use_color=True,
        scene=False, inline_portraits=True,
    )
    assert "▀" not in frame
    overlays = _portrait_overlays(frame, left=14, width=174, compact=False)
    assert overlays.count("\x1b7") == overlays.count("\x1b8") == 4
    for up in (19, 15, 11, 7):
        assert f"\x1b7\r\x1b[{up}A\x1b[56C" in overlays


def test_three_column_banner_places_modes_between_trainer_and_status() -> None:
    for width in (105, 130, 174):
        frame = _render_banner_text(
            STATUS, width=width, terminal_lines=47,
            use_color=False, scene=False, inline_portraits=True,
        )
        header = frame.splitlines()[3]
        assert header.index("Welcome") < header.index("Choose your mode")
        status_row = next(row for row in frame.splitlines() if "PIKA STATUS" in row)
        assert status_row.index("PIKA STATUS") > header.index("Choose your mode")


def test_wide_banner_aligns_all_three_headings() -> None:
    frame = _render_banner_text(
        STATUS, width=174, terminal_lines=47,
        use_color=False, scene=False, inline_portraits=True,
        large_titles=True,
    )
    rows = frame.splitlines()
    assert all(title in rows[4] for title in (
        "Welcome", "Choose your mode · 1-4", "▣ PIKA STATUS",
    ))
    assert "Welcome back!" not in frame
    assert "Welcome" not in rows[5]


def test_iterm_title_overlays_fill_reserved_two_row_headings() -> None:
    frame = _render_banner_text(
        STATUS, width=174, terminal_lines=47, use_color=True,
        scene=False, inline_portraits=True, large_titles=True,
    )
    rows = frame.splitlines()
    assert len(rows) == 25
    assert "Pikachu CLI" in rows[1] and "╠" not in rows[2]
    overlays = _title_overlays(frame, left=14, width=174)
    assert overlays.count("File=inline=1") == 5
    assert overlays.count("height=2") == 5
    assert overlays.count("\x1b7") == overlays.count("\x1b8") == 5


def test_trainer_and_bubble_use_the_supplied_images_and_restore_cursor() -> None:
    frame = _render_banner_text(
        STATUS, width=174, terminal_lines=47, use_color=True,
        scene=False, inline_portraits=True,
    )
    overlays = _trainer_overlays(frame, left=14, width=174, compact=False)
    assert overlays.count("\x1b7") == overlays.count("\x1b8") == 2
    for name in ("trainer_ash.png", "trainer_bubble.png"):
        with Image.open(BACKGROUND.with_name(name)) as image:
            assert image.mode == "RGBA"
            assert image.getpixel((0, 0))[3] == 0
    assert "width=16;height=11" in overlays
    assert "width=16;height=5" in overlays


def test_mode_badges_share_one_terminal_column() -> None:
    for width in (116, 174):
        frame = _render_banner_text(
            STATUS, width=width, terminal_lines=47,
            use_color=False, scene=False, inline_portraits=True,
        )
        badge_rows = [row for row in frame.splitlines() if any(
            name in row for name in ("[ NORMAL", "[  PLAN", "[ LIGHT")
        )]
        assert len(badge_rows) == 4
        assert len({_terminal_display_width(row.split("[", 1)[0]) for row in badge_rows}) == 1


def test_agent_portraits_render_from_four_distinct_cutouts() -> None:
    portraits = [_portrait_pixels(agent, 12, 8) for agent in ("1", "2", "3", "4")]
    assert len({pixels for pixels in portraits}) == 4
    assert all(len(pixels) == 8 and len(pixels[0]) == 12 for pixels in portraits)
    assert any(red > 180 and blue > 100 and green < 210 for row in portraits[1] for red, green, blue in row)
    for agent in ("1", "2", "3", "4"):
        line = _portrait_line(agent, 0, use_color=True, width=12, height=8)
        assert line.count("▀") == 12
        assert "\033[38;2;" in line and ";48;2;" in line


def test_tall_welcome_frame_fits_four_portrait_rows_per_agent() -> None:
    frame = _render_banner_text(
        STATUS, width=174, terminal_lines=47, use_color=True, scene=False
    )
    assert len(frame.splitlines()) == 23
    assert all(name in frame for name in ("Lucario", "Jigglypuff", "Psyduck", "Pikachu"))


def test_actual_terminal_size_wins_over_stale_shell_dimensions() -> None:
    class TerminalOutput(io.StringIO):
        def fileno(self) -> int:
            return 7

    with (
        patch.dict(os.environ, {"COLUMNS": "80", "LINES": "24"}),
        patch("sys.stdout", TerminalOutput()),
        patch("os.get_terminal_size", return_value=os.terminal_size((140, 40))) as actual,
    ):
        assert terminal_size() == os.terminal_size((140, 40))
    actual.assert_called_once_with(7)


def test_actual_token_usage_is_reported_after_model_call_and_turn(capsys) -> None:
    event = AgentEvent(
        run_id="test-run",
        type=AgentEventType.MODEL_COMPLETED,
        usage=ModelUsage(input_tokens=123, output_tokens=45, total_tokens=168),
    )
    print_agent_event(event)
    with patch("shutil.get_terminal_size", return_value=os.terminal_size((120, 40))):
        print_live_usage(
            context_used=6_000,
            context_window=100_000,
            run_tokens=168,
            session_tokens=505,
        )
    text = capsys.readouterr().out
    assert "输入 123 · 输出 45 · 合计 168 tokens" in text
    assert "Context ≈6,000/100,000" in text
    assert "本轮 168 · 本次 CLI 会话累计 505" in text


def test_dispatch_usage_reaches_interactive_caller(capsys) -> None:
    class Service:
        received_prompt: str | None = None

        async def dispatch(self, **kwargs):
            self.received_prompt = kwargs.get("additional_system_prompt")
            return SimpleNamespace(
                result=SimpleNamespace(
                    messages=["updated"],
                    steps=1,
                    usage=ModelUsage(input_tokens=123, output_tokens=45, total_tokens=168),
                    stop_reason=SimpleNamespace(value="completed"),
                    tool_calls=(),
                    final_message=SimpleNamespace(content=None),
                    ok=True,
                ),
                run=SimpleNamespace(
                    stop_reason=None,
                    status=SimpleNamespace(value="completed"),
                ),
            )

    conversation = SimpleNamespace(id="conversation-id", title="existing")
    history = []
    service = Service()
    success, current, tokens = asyncio.run(_send_message(
        conversation_service=service,
        conversation_store=None,
        conversation=conversation,
        provider="deepseek",
        history=history,
        content="hello",
        model="deepseek-flash",
        mode=AgentMode.DEFAULT,
        additional_system_prompt="编码优先",
    ))
    assert success and current is conversation
    assert history == ["updated"]
    assert tokens == 168
    assert service.received_prompt == "编码优先"
    assert "168 tokens" in capsys.readouterr().out


def test_resize_redraws_banner_at_next_safe_input() -> None:
    answers = iter(("", "/exit"))

    async def read_input(*args):
        return next(answers)

    app = SimpleNamespace(
        provider="deepseek",
        model="deepseek-flash",
        context_manager=SimpleNamespace(context_window_for=lambda *args: 100_000),
        tool_registry=SimpleNamespace(definitions_for_mode=lambda *args: ()),
    )
    with (
        patch("app.interfaces.cli.chat.terminal_size", side_effect=(
            os.terminal_size((100, 40)),
            os.terminal_size((180, 40)),
            os.terminal_size((180, 40)),
        )),
        patch("app.interfaces.cli.chat.asyncio.to_thread", side_effect=read_input),
        patch("app.interfaces.cli.chat.default_token_estimator", return_value=SimpleNamespace(
            estimate_request=lambda *args, **kwargs: 6_000,
        )),
        patch("app.interfaces.cli.chat.print_banner") as redraw,
    ):
        result = asyncio.run(_run_interactive(
            app=app,
            conversation=None,
            history=[],
            system_prompt=None,
            mode=AgentMode.DEFAULT,
        ))
    assert result == 0
    redraw.assert_called_once()
    assert ("Context", "6,000 / 100,000") in redraw.call_args.kwargs["status"]


def test_numbered_agents_switch_mode_and_run_preference() -> None:
    answers = iter(("2", "写一首诗", "3", "先规划", "4", "轻量执行", "1", "修复代码", "/exit"))
    sent: list[tuple[AgentMode, str | None]] = []

    async def read_input(*args):
        return next(answers)

    async def send_message(**kwargs):
        sent.append((kwargs["mode"], kwargs["additional_system_prompt"]))
        return True, kwargs["conversation"], 0

    async def handle_command(content, **kwargs):
        return False, kwargs["conversation"], kwargs["history"]

    app = SimpleNamespace(
        provider="deepseek",
        model="deepseek-flash",
        context_manager=SimpleNamespace(context_window_for=lambda *args: 100_000),
        tool_registry=SimpleNamespace(definitions_for_mode=lambda *args: ()),
        conversation_service=object(),
        conversation_store=object(),
    )
    with (
        patch("app.interfaces.cli.chat.terminal_size", return_value=os.terminal_size((202, 47))),
        patch("app.interfaces.cli.chat.asyncio.to_thread", side_effect=read_input),
        patch("app.interfaces.cli.chat.default_token_estimator", return_value=SimpleNamespace(
            estimate_request=lambda *args, **kwargs: 6_000,
        )),
        patch("app.interfaces.cli.chat.print_banner") as redraw,
        patch("app.interfaces.cli.chat._send_message", side_effect=send_message),
        patch("app.interfaces.cli.chat._handle_command", side_effect=handle_command),
    ):
        result = asyncio.run(_run_interactive(
            app=app, conversation=object(), history=[], system_prompt=None,
            mode=AgentMode.DEFAULT,
        ))
    assert result == 0
    assert [mode for mode, _ in sent] == [
        AgentMode.DEFAULT, AgentMode.PLAN, AgentMode.LIGHT, AgentMode.DEFAULT,
    ]
    assert sent[0][1] and "创作" in sent[0][1]
    assert sent[1][1] is None and sent[2][1] is None
    assert sent[3][1] and "编码" in sent[3][1]
    redraw.assert_not_called()


def test_slash_mode_switches_do_not_repeat_the_welcome_banner(capsys) -> None:
    answers = iter(("/light", "/plan", "/light off", "/plan off", "/exit"))

    async def read_input(*args):
        return next(answers)

    app = SimpleNamespace(provider="deepseek", model="deepseek-flash")
    with (
        patch("app.interfaces.cli.chat.terminal_size", return_value=os.terminal_size((202, 47))),
        patch("app.interfaces.cli.chat.asyncio.to_thread", side_effect=read_input),
        patch("app.interfaces.cli.chat.print_banner") as redraw,
    ):
        result = asyncio.run(_run_interactive(
            app=app, conversation=object(), history=[], system_prompt=None,
            mode=AgentMode.DEFAULT,
        ))
    assert result == 0
    redraw.assert_not_called()
    output = capsys.readouterr().out
    assert "已进入 LIGHT MODE" in output
    assert "已进入 PLAN MODE" in output
    assert "已退出 LIGHT MODE" in output
    assert "已退出 PLAN MODE" in output

"""Aider Polyglot 基准评测脚本（sidekick 版本）。

使用方法：
    python run_polyglot_eval.py --exercise book-store
    python run_polyglot_eval.py --language python --limit 5
    python run_polyglot_eval.py --language python   # 跑全部 python 题
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 触发跨模块前向引用重建（必须先 import Application，再调 rebuild_models，
# 因为 model/__init__.py 中有循环 import）。
from app.application import Application
from app.model.config import rebuild_models

rebuild_models()

from app.agent.events import AgentEvent, AgentEventHandler
from app.conversation.service import ConversationSource, TriggerContext
from app.model.config import ModelProvider
from app.run.outcome_store import RunOutcome, RunOutcomeStore
from app.tools.approval import AutoApproveGate


REPO_ROOT = Path("/Users/gaojiayi/projects/sidekick-main")
BACKEND = REPO_ROOT / "backend"
POLYGLOT = REPO_ROOT / "eval" / "polyglot-benchmark-main"

import logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

# 语言 → (测试文件名后缀, 跑测试的命令)
LANG_CONFIG = {
    "python": (".py", "python -m pytest -q {test}"),
    "go": (".go", "go test -count=1"),
    "javascript": (".js", "npx jest --silent"),
    "rust": (".rs", "cargo test --quiet"),
    "java": (".java", "./gradlew test --quiet"),
    "cpp": (
        ".cpp",
        'mkdir -p build && cd build && '
        'cmake -DEXERCISM_RUN_ALL_TESTS=1 -G "Unix Makefiles" .. && make',
    ),
}


@dataclass
class TaskResult:
    exercise: str
    language: str
    passed_first: bool
    passed: bool
    edit_attempts: int
    final_stdout: str
    final_stderr: str
    detail: str
    initial_run_id: str | None = None
    feedback_run_id: str | None = None


class PolyglotCollector(AgentEventHandler):
    """监听工具调用,统计 write/edit 调用,并记录是否真的跑过测试。"""

    # 命令前缀白名单:出现这些就视为调用了测试
    PYTEST_PREFIXES = ("pytest", "python -m pytest", "python3 -m pytest")
    # C++/Go/Rust/JS/Java:通用"测试"命令关键词
    GENERIC_TEST_KEYWORDS = ("test",)

    def __init__(self, language: str = "python") -> None:
        self.language = language
        self.write_calls = 0
        self.tool_calls: list[tuple[str, str]] = []
        self.all_events: list[AgentEvent] = []
        self.test_command_calls: list[str] = []  # 记录每次疑似测试命令
        self.tested = False  # 是否真的调过一次测试命令

    async def handle(self, event: AgentEvent) -> None:  # noqa: D401
        # 记录全部事件,便于排查 hook 是否真的执行了
        self.all_events.append(event)
        name = None
        args: Any = None
        if event.tool_call is not None:
            name = event.tool_call.name
            args = event.tool_call.arguments
        elif event.tool_result is not None:
            name = event.tool_result.tool_name
        if name is not None:
            args_preview = str(args)[:80]
            self.tool_calls.append((name, args_preview))
            if name in {"write_file", "edit_file", "apply_patch"}:
                self.write_calls += 1
            # run_shell_command 视为调用测试的入口
            if name == "run_shell_command" and args:
                cmd = args.get("command") if isinstance(args, dict) else None
                if cmd and self._looks_like_test(cmd):
                    self.test_command_calls.append(cmd.strip()[:200])
                    self.tested = True

    def _looks_like_test(self, cmd: str) -> bool:
        cmd_l = cmd.strip().lower()
        if self.language == "python":
            return any(cmd_l.startswith(p) for p in self.PYTEST_PREFIXES) or (
                "pytest" in cmd_l
            )
        # 其它语言:见名知意即可(包 test / spec / cargo test 等)
        return any(kw in cmd_l for kw in self.GENERIC_TEST_KEYWORDS)


def _is_test_file(language: str, name: str) -> bool:
    """判断文件名是否是该语言的测试文件。"""
    if language == "python":
        return name.endswith("_test.py")
    if language == "go":
        return name.endswith("_test.go")
    if language == "javascript":
        return name.endswith(".spec.js") or name.endswith(".test.js")
    if language == "java":
        return name.endswith("Test.java")
    if language == "cpp":
        return name.endswith("_test.cpp")
    if language == "rust":
        return False  # rust 测试在 tests/ 目录，由 _copy_tree 单独排除
    return False


def _copy_tree(
    src: Path,
    dst: Path,
    *,
    language: str,
    exclude_tests: bool,
) -> None:
    """递归拷贝 src 到 dst；跳过隐藏文件/目录，可选跳过测试文件。"""
    for item in src.rglob("*"):
        rel = item.relative_to(src)
        parts = rel.parts
        if any(p.startswith(".") for p in parts):
            continue
        # 这些是本地测试运行产生的缓存，可能包含测试模块的字节码。
        # 无论是 workspace 还是评分目录，都不能把它们带进去。
        if any(p in {"__pycache__", ".pytest_cache"} for p in parts):
            continue
        if item.suffix == ".pyc":
            continue
        if item.is_dir():
            continue
        if exclude_tests:
            if _is_test_file(language, item.name):
                continue
            if language == "cpp" and "test" in parts:
                continue
            if language == "rust" and "tests" in parts:
                continue
            # 评测过程中生成的辅助验证脚本不属于题目实现。
            if item.name.startswith("verify_"):
                continue
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def _stub_candidates(language: str, exercise_dir: Path) -> list[Path]:
    """按语言找出「实现文件」候选（排除测试文件）。"""
    if language == "python":
        return sorted(
            p for p in exercise_dir.iterdir()
            if p.is_file() and p.suffix == ".py" and not _is_test_file(language, p.name)
        )
    if language == "go":
        return sorted(
            p for p in exercise_dir.iterdir()
            if p.is_file() and p.suffix == ".go" and not _is_test_file(language, p.name)
        )
    if language == "javascript":
        return sorted(
            p for p in exercise_dir.iterdir()
            if p.is_file()
            and p.suffix == ".js"
            and not _is_test_file(language, p.name)
            and p.name != "babel.config.js"  # babel.config.js 是构建配置，不是解法文件
        )
    if language == "java":
        return sorted(
            p for p in exercise_dir.glob("src/main/java/**/*.java")
            if p.is_file() and not _is_test_file(language, p.name)
        )
    if language == "rust":
        for name in ("src/lib.rs", "src/main.rs"):
            p = exercise_dir / name
            if p.is_file():
                return [p]
        return []
    if language == "cpp":
        return sorted(
            p for p in exercise_dir.iterdir()
            if p.is_file()
            and p.suffix in (".cpp", ".cc")
            and not _is_test_file(language, p.name)
        )
    return []


def _load_prompt(language: str, exercise_dir: Path) -> tuple[str, str, str]:
    """读 instructions.md 和 stub，返回 (instructions, stub, stub_relpath)。"""

    instructions_path = exercise_dir / ".docs" / "instructions.md"
    if not instructions_path.exists():
        raise FileNotFoundError(f"缺少 instructions.md: {instructions_path}")
    instructions = instructions_path.read_text(encoding="utf-8")

    candidates = _stub_candidates(language, exercise_dir)
    if not candidates:
        raise FileNotFoundError(f"找不到 stub 文件: {exercise_dir}")
    stub_path = candidates[0]
    stub = stub_path.read_text(encoding="utf-8")
    return instructions, stub, stub_path.relative_to(exercise_dir).as_posix()


def _run_tests(
    language: str,
    exercise_dir: Path,
    solution_stub: Path,
) -> tuple[bool, str, str]:
    """把真实测试与模型写的实现放到评分目录，独立跑测试打分。

    盲测：模型看不到测试；评分在 workspace 之外的 .scoring 目录进行。
    """
    _, cmd_template = LANG_CONFIG[language]

    # 评分目录放在 workspace 之外的临时目录，跑完即删：模型在任何一轮都看不到测试。
    with tempfile.TemporaryDirectory(prefix="polyglot-scoring-") as tmp:
        # C++ 的 CMakeLists.txt 会根据当前目录名推导 exercise 名称，
        # 所以评分目录必须保留原 exercise 名，不能直接使用随机临时目录名。
        scoring_dir = Path(tmp) / exercise_dir.name
        scoring_dir.mkdir()

        # 1) 题目目录整体拷进评分目录（含真实测试与项目配置；跳过隐藏目录）
        _copy_tree(exercise_dir, scoring_dir, language=language, exclude_tests=False)

        # 2) 覆盖模型写的实现（workspace → scoring，跳过测试文件与隐藏文件）
        _copy_tree(solution_stub.parent, scoring_dir, language=language, exclude_tests=True)

        # javascript：启用 xtest 用例。Exercism JS 测试默认大量用 xtest 跳过，
        # 若不替换成 test，jest 只跑第一个用例，通过率会虚高。
        if language == "javascript":
            for spec in scoring_dir.rglob("*.spec.js"):
                text = spec.read_text(encoding="utf-8")
                spec.write_text(text.replace("xtest(", "test("), encoding="utf-8")

        # 模型可能在 workspace 中执行过 CMake，复制过来的缓存记录了旧路径；
        # 评分时必须从干净的 build 目录重新配置。
        if language == "cpp":
            shutil.rmtree(scoring_dir / "build", ignore_errors=True)

        cmd = cmd_template
        if language == "python":
            test_file = next((p for p in scoring_dir.glob("*_test.py")), None)
            if test_file is None:
                return False, "", "找不到测试文件"
            cmd = cmd_template.format(test=test_file.name)

        # javascript：先装依赖（jest + babel）再跑测试
        if language == "javascript":
            install = subprocess.run(
                "npm install --silent",
                shell=True,
                cwd=str(scoring_dir),
                capture_output=True,
                text=True,
                timeout=300,
            )
            if install.returncode != 0:
                return False, install.stdout, install.stderr

        timeout = 300 if language != "python" else 180
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=str(scoring_dir),
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", f"测试超时 (>{timeout}s)"
        except Exception as exc:  # noqa: BLE001
            return False, "", str(exc)


_LANG_CODE_HINT = {
    "python": "python",
    "go": "go",
    "javascript": "javascript",
    "rust": "rust",
    "java": "java",
    "cpp": "cpp",
}


def build_user_prompt(
    instructions: str,
    stub_relpath: str,
    stub: str,
    language: str,
) -> str:
    hint = _LANG_CODE_HINT.get(language, "")
    return f"""你需要完成一道 Exercism 编程题。请按以下步骤操作：

1. 阅读下面的题目描述。
2. 在 **{stub_relpath}** 中实现对应功能，正确处理题目描述中的所有规则与边界情况。
3. 可以用 run_shell_command 跑自测命令验证逻辑，但正式的测试文件由评测方隐藏运行，你无法查看。
4. 完成后说明你做了哪些修改、以及你认为有哪些边界情况需要特别注意。

题目描述：

{instructions}

当前 stub 文件 {stub_relpath} 的内容：

```{hint}
{stub}
```

注意：
- 只实现 {stub_relpath}；如需可以新建辅助文件，但不要创建或修改任何测试文件。
- 评分时评测方会独立运行测试，你的代码只需正确实现题目要求的逻辑。
"""


def build_feedback_prompt(stdout: str, stderr: str, stub_relpath: str) -> str:
    """失败后把测试输出喂回给模型，请求一轮修复（不暗示可改测试）。"""

    # C++ 的 cmake/make 把编译错误输出到 stderr、进度输出到 stdout，
    # 两者都要喂回，否则模型只看到进度、看不到 error/FAILED。
    output = (stdout.rstrip() + "\n" + stderr.rstrip()).strip()
    if not output:
        output = "（评测方未提供测试输出）"
    output = output[-60000:]  # 截断，避免超长输出撑爆上下文
    return (
        "你的实现没有通过评测方的测试。测试是正确的，不要修改测试。\n"
        f"请根据下面的测试错误信息，修复 {stub_relpath} 里的代码"
        "（如需可以新建辅助文件，但不要创建或修改任何测试文件）。\n\n"
        f"测试错误信息：\n{output}"
    )


def find_exercises(language: str, limit: int | None) -> list[Path]:
    base = POLYGLOT / language / "exercises" / "practice"
    if not base.exists():
        raise FileNotFoundError(f"目录不存在: {base}")
    items = sorted([p for p in base.iterdir() if p.is_dir()])
    if limit:
        items = items[:limit]
    return items


async def _build_app(
    args: argparse.Namespace,
    exercise_dir: Path,
) -> Application:
    """为单个 exercise 构造独立的 Application + 独立 workspace（盲测）。

    workspace 里放题目目录的全部非测试文件（stub + 项目结构），但不放测试
    文件与 .meta/.docs，模型既看不到测试、也看不到参考解。评分阶段单独把
    模型写的实现拷到评分目录配真实测试跑。
    """
    from app.agent.budget import RunBudgetConfig

    workspace = REPO_ROOT / "workspace" / "eval" / args.language / exercise_dir.name
    if workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    # 盲测：整目录拷进 workspace，但排除测试文件与 .docs/.meta 等隐藏目录。
    _copy_tree(exercise_dir, workspace, language=args.language, exclude_tests=True)

    return Application(
        provider=ModelProvider(args.provider),
        model=args.model,
        database=args.database,
        tasks_dir=args.tasks_dir,
        workspace_root=workspace,
        # 盲测：只放行 workspace 内的 shell 自测命令，网络类工具仍走默认拒绝。
        # approval_gate=AutoApproveGate(approve_tool_names=()),
        max_steps=args.max_steps,
        max_tool_rounds=args.max_tool_rounds,
        max_output_tokens=4096,
        run_budget_config=RunBudgetConfig(
            hard_model_calls=args.max_model_calls,
            finalization_model_calls=max(1, args.max_model_calls - 5),
            warning_model_calls=max(1, args.max_model_calls - 10),
            hard_tokens=args.max_chargeable_tokens,
            finalization_tokens=max(1, args.max_chargeable_tokens - 20000),
            warning_tokens=max(1, args.max_chargeable_tokens - 60000),
        ),
        ace_enabled=getattr(args, "ace", False),
        ace_playbook_path=(
            REPO_ROOT / ".database" / "ace" / f"{args.language}_playbook.txt"
        ),
    )


async def _dispatch_turn(
    app: Application,
    conversation: Any,
    content: str,
    collector: PolyglotCollector,
    *,
    source: ConversationSource = ConversationSource.MANUAL,
) -> tuple[str, str | None, str | None]:
    """发一轮消息，返回 (final_message, error, run_id)。"""

    try:
        result = await app.conversation_service.dispatch(
            conversation_id=conversation.id,
            content=content,
            trigger=TriggerContext(source=source),
            event_handler=collector,
        )
        final = ""
        if result.result is not None:
            final = result.result.final_message.content or ""
        return final, None, result.run.id
    except Exception as exc:  # noqa: BLE001
        import traceback
        return "", f"{exc}\n{traceback.format_exc()}", None


async def run_one(
    language: str,
    exercise_dir: Path,
    *,
    args: argparse.Namespace,
) -> TaskResult:
    instructions, stub, stub_relpath = _load_prompt(language, exercise_dir)
    user_prompt = build_user_prompt(instructions, stub_relpath, stub, language)

    app = await _build_app(args, exercise_dir)
    await app.start()
    outcome_store = RunOutcomeStore(args.database or app.database)
    await outcome_store.initialize()
    initial_run_id: str | None = None
    feedback_run_id: str | None = None
    try:
        collector = PolyglotCollector(language=language)
        conversation = await app.conversation_store.create()
        solution_stub = app.workspace_root / stub_relpath

        # 第一轮（盲测）：模型在只有 stub 的独立 workspace 里实现
        final_message, dispatch_error, initial_run_id = await _dispatch_turn(
            app,
            conversation,
            user_prompt,
            collector,
            source=ConversationSource.EVAL_INITIAL,
        )

        passed_first = False
        passed = False
        stdout = stderr = ""
        if dispatch_error is None and solution_stub.exists():
            # 评分：把模型写的 stub 拷到评分目录，配真实测试文件独立跑
            passed, stdout, stderr = _run_tests(language, exercise_dir, solution_stub)
            passed_first = passed
            if initial_run_id is not None:
                await outcome_store.save(
                    run_id=initial_run_id,
                    language=language,
                    exercise=exercise_dir.name,
                    phase="initial",
                    outcome=RunOutcome.PASSED if passed else RunOutcome.FAILED,
                    stdout=stdout,
                    stderr=stderr,
                )

                # ACE learns from the blind initial trajectory only after the
                # external evaluator has produced its authoritative result.
                # The feedback repair run below is intentionally excluded from
                # ACE learning because it has already seen hidden-test output.
                if initial_run_id is not None and app.ace is not None:
                    initial_result = app.run_manager.result(initial_run_id)
                    if initial_result is not None:
                        evaluator_feedback = (
                            f"phase=initial\n"
                            f"stdout:\n{stdout[-60000:]}\n"
                            f"stderr:\n{stderr[-60000:]}"
                        )
                        await app.ace.reflect(
                            user_input=user_prompt,
                            result=initial_result,
                            selection=app.ace.resolve_selection(
                                app.ace.select(user_prompt),
                                (
                                    initial_result.messages[-1].content
                                    if initial_result.messages
                                    else initial_result.content
                                ),
                            ),
                            evaluator_feedback=evaluator_feedback,
                            outcome="passed" if passed else "failed",
                        )

            # feedback 模式：失败后把测试输出喂回，让模型在同一会话里做一轮修复
            if args.feedback and not passed:
                feedback = build_feedback_prompt(stdout, stderr, stub_relpath)
                final_message, dispatch_error, feedback_run_id = await _dispatch_turn(
                    app, conversation, feedback, collector,
                    source=ConversationSource.EVAL_FEEDBACK,
                )
                if dispatch_error is None and solution_stub.exists():
                    passed, stdout, stderr = _run_tests(language, exercise_dir, solution_stub)
                    if feedback_run_id is not None:
                        await outcome_store.save(
                            run_id=feedback_run_id,
                            language=language,
                            exercise=exercise_dir.name,
                            phase="feedback",
                            outcome=RunOutcome.PASSED if passed else RunOutcome.FAILED,
                            stdout=stdout,
                            stderr=stderr,
                        )
    finally:
        await app.close()

    if dispatch_error is not None:
        return TaskResult(
            exercise=exercise_dir.name,
            language=language,
            passed_first=False,
            passed=False,
            edit_attempts=collector.write_calls,
            final_stdout="",
            final_stderr="",
            detail=f"dispatch 异常: {dispatch_error}",
            initial_run_id=initial_run_id,
            feedback_run_id=feedback_run_id,
        )

    solution_stub = app.workspace_root / stub_relpath
    if not solution_stub.exists():
        return TaskResult(
            exercise=exercise_dir.name,
            language=language,
            passed_first=False,
            passed=False,
            edit_attempts=collector.write_calls,
            final_stdout="",
            final_stderr="",
            detail="模型未写出 stub 文件",
            initial_run_id=initial_run_id,
            feedback_run_id=feedback_run_id,
        )

    tool_summary = ", ".join(f"{n}({a})" for n, a in collector.tool_calls) or "<no tool calls>"
    event_types = [e.type.value for e in collector.all_events]
    test_summary = (
        f"模型调用过测试命令 {len(collector.test_command_calls)} 次:{collector.test_command_calls}"
        if collector.tested
        else "模型未调用测试命令（盲测模式，无测试文件）"
    )
    detail = (
        f"tested={collector.tested} ({test_summary})\n"
        + (
            "通过"
            if passed
            else (
                f"未通过\n--- tools ---\n{tool_summary}\n--- events ---\n{event_types[:80]}"
                f"\n--- final ---\n{final_message[:400]}"
                f"\n--- stdout ---\n{stdout[-500:]}\n--- stderr ---\n{stderr[-500:]}"
            )
        )
    )
    return TaskResult(
        exercise=exercise_dir.name,
        language=language,
        passed_first=passed_first,
        passed=passed,
        edit_attempts=collector.write_calls,
        final_stdout=stdout,
        final_stderr=stderr,
        detail=detail,
        initial_run_id=initial_run_id,
        feedback_run_id=feedback_run_id,
    )


async def main_async(args: argparse.Namespace) -> int:
    languages = (
        list(LANG_CONFIG.keys())
        if args.language == "all"
        else [args.language]
    )

    results: list[TaskResult] = []
    for lang in languages:
        # 默认按 language 分库；显式指定 --database 则全部写同一个库。
        lang_args = args
        if args.database is None:
            lang_args = argparse.Namespace(**vars(args))
            lang_args.database = str(BACKEND / ".database" / f"polyglot-{lang}.db")
        exercises = find_exercises(lang, args.limit)
        if args.exercise:
            exercises = [e for e in exercises if e.name == args.exercise]
            if not exercises:
                print(f"[{lang}] 找不到 exercise: {args.exercise}", file=sys.stderr)
                continue

        print(f"评测范围: {lang} · {len(exercises)} 题\n")
        for i, ex_dir in enumerate(exercises, 1):
            print(f"[{lang} {i}/{len(exercises)}] {ex_dir.name} ... ", end="", flush=True)
            result = await run_one(lang, ex_dir, args=lang_args)
            results.append(result)
            if args.report:
                _write_report(results, args.report)
            status = "✓" if result.passed else "✗"
            print(f"{status}  (edit={result.edit_attempts}) {result.detail[:100]}")

        lang_results = [r for r in results if r.language == lang]
        lang_passed_1 = sum(r.passed_first for r in lang_results)
        lang_passed_2 = sum(r.passed for r in lang_results)
        print(
            f"[{lang}] pass@1 {lang_passed_1} / {len(exercises)} = "
            f"{lang_passed_1 / len(exercises) * 100:.1f}%；"
            f"pass@2 {lang_passed_2} / {len(exercises)} = "
            f"{lang_passed_2 / len(exercises) * 100:.1f}%\n"
        )

    passed_1 = sum(r.passed_first for r in results)
    passed_2 = sum(r.passed for r in results)
    total = len(results)
    print(f"\n=== 汇总 ===")
    if total:
        print(f"pass@1 {passed_1} / {total} = {passed_1 / total * 100:.1f}%")
        print(f"pass@2 {passed_2} / {total} = {passed_2 / total * 100:.1f}%")
    else:
        print("没有可统计的结果")

    if args.report:
        _write_report(results, args.report)
        print(f"报告已写入 {REPO_ROOT / args.report}")

    return 0 if passed_2 == total else 1


def _write_report(results: list[TaskResult], report_name: str) -> None:
    """Incrementally persist completed exercises so interruptions lose no results."""

    report = REPO_ROOT / report_name
    report.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "exercise": r.exercise,
            "language": r.language,
            "passed_at_1": r.passed_first,
            "passed_at_2": r.passed,
            "passed": r.passed,
            "initial_run_id": r.initial_run_id,
            "feedback_run_id": r.feedback_run_id,
            "edit_attempts": r.edit_attempts,
            "stdout_tail": r.final_stdout,
            "stderr_tail": r.final_stderr,
            "detail": r.detail,
        }
        for r in results
    ]
    temporary = report.with_suffix(report.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
    temporary.replace(report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--language",
        default="python",
        help="语言（python/go/javascript/rust/java/cpp），或 'all' 跑全部",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--exercise", default=None, help="只跑单个 exercise 名")
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--max-tool-rounds", type=int, default=30)
    parser.add_argument(
        "--max-model-calls",
        type=int,
        default=80,
        help="RunBudgetConfig.hard_model_calls 上限",
    )
    parser.add_argument(
        "--max-chargeable-tokens",
        type=int,
        default=200000,
        help="RunBudgetConfig.hard_tokens 上限",
    )
    parser.add_argument(
        "--database",
        default=None,
        help="数据库路径；不指定则按 language 自动生成 polyglot-{language}.db",
    )
    parser.add_argument(
        "--tasks-dir",
        default=str(BACKEND / f"tasks-{Path(tempfile.gettempdir()).name}"),
    )
    parser.add_argument("--report", default=None)
    parser.add_argument(
        "--feedback",
        action="store_true",
        help="失败后把测试输出喂回模型做一轮修复（feedback 模式）。默认关闭（非 feedback 盲测）。",
    )
    parser.add_argument(
        "--ace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用 ACE playbook；默认关闭，可用 --no-ace 显式关闭。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()

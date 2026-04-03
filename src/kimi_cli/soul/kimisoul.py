from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast
from uuid import uuid4

import kosong
import tenacity
from kosong import StepResult
from kosong.chat_provider import (
    APIConnectionError,
    APIEmptyResponseError,
    APIStatusError,
    APITimeoutError,
    RetryableChatProvider,
)
from kosong.message import Message
from tenacity import RetryCallState, retry_if_exception, stop_after_attempt, wait_exponential_jitter

from kimi_cli.approval_runtime import (
    ApprovalSource,
    get_current_approval_source_or_none,
    reset_current_approval_source,
    set_current_approval_source,
)
from kimi_cli.background import build_active_task_snapshot
from kimi_cli.hooks.engine import HookEngine
from kimi_cli.llm import ModelCapability
from kimi_cli.notifications import (
    NotificationView,
    build_notification_message,
    extract_notification_ids,
)
from kimi_cli.skill import Skill, read_skill_text
from kimi_cli.skill.flow import Flow, FlowEdge, FlowNode, parse_choice
from kimi_cli.soul import (
    LLMNotSet,
    LLMNotSupported,
    MaxStepsReached,
    Soul,
    StatusSnapshot,
    get_wire_or_none,
    wire_send,
)
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.compaction import (
    CompactionResult,
    SimpleCompaction,
    estimate_text_tokens,
    should_auto_compact,
)
from kimi_cli.soul.context import Context
from kimi_cli.soul.dynamic_injection import (
    DynamicInjection,
    DynamicInjectionProvider,
    normalize_history,
)
from kimi_cli.soul.dynamic_injections.plan_mode import PlanModeInjectionProvider
from kimi_cli.soul.dynamic_injections.yolo_mode import YoloModeInjectionProvider
from kimi_cli.soul.message import check_message, system, system_reminder, tool_result_to_message
from kimi_cli.soul.slash import registry as soul_slash_registry
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.dmail import NAME as SendDMail_NAME
from kimi_cli.tools.utils import ToolRejectedError
from kimi_cli.utils.logging import logger
from kimi_cli.utils.slashcmd import SlashCommand, parse_slash_command_call
from kimi_cli.utils.turns import is_real_user_turn_start_message
from kimi_cli.wire.file import WireFile
from kimi_cli.wire.types import (
    CompactionBegin,
    CompactionEnd,
    ContentPart,
    FollowUpInput,
    MCPLoadingBegin,
    MCPLoadingEnd,
    QuestionItem,
    QuestionNotSupported,
    QuestionOption,
    QuestionRequest,
    StatusUpdate,
    SteerInput,
    StepBegin,
    StepInterrupted,
    TextPart,
    ToolResult,
    TurnBegin,
    TurnEnd,
)

if TYPE_CHECKING:

    def type_check(soul: KimiSoul):
        _: Soul = soul


SKILL_COMMAND_PREFIX = "skill:"
FLOW_COMMAND_PREFIX = "flow:"
DEFAULT_MAX_FLOW_MOVES = 1000

TURN_END_QUESTION_DETECTOR_PROMPT = (
    Path(__file__).resolve().parent.parent / "prompts" / "turn_end_question_detector.md"
).read_text()


type StepStopReason = Literal["no_tool_calls", "tool_rejected"]


@dataclass(frozen=True, slots=True)
class StepOutcome:
    stop_reason: StepStopReason
    assistant_message: Message


type TurnStopReason = StepStopReason


@dataclass(frozen=True, slots=True)
class TurnOutcome:
    stop_reason: TurnStopReason
    final_message: Message | None
    step_count: int


@dataclass(frozen=True, slots=True)
class TurnEndQuestionOption:
    label: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class TurnEndQuestionItem:
    question: str
    options: tuple[TurnEndQuestionOption, ...]


@dataclass(frozen=True, slots=True)
class TurnEndQuestionDetection:
    has_question: bool
    questions: tuple[TurnEndQuestionItem, ...]


class KimiSoul:
    """The soul of Kimi Code CLI."""

    def __init__(
        self,
        agent: Agent,
        *,
        context: Context,
    ):
        """
        Initialize the soul.

        Args:
            agent (Agent): The agent to run.
            context (Context): The context of the agent.
        """
        self._agent = agent
        self._runtime = agent.runtime
        self._denwa_renji = agent.runtime.denwa_renji
        self._approval = agent.runtime.approval
        self._context = context
        self._loop_control = agent.runtime.config.loop_control
        self._compaction = SimpleCompaction()  # TODO: maybe configurable and composable

        for tool in agent.toolset.tools:
            if tool.name == SendDMail_NAME:
                self._checkpoint_with_user_message = True
                break
        else:
            self._checkpoint_with_user_message = False

        self._steer_queue: asyncio.Queue[str | list[ContentPart]] = asyncio.Queue()
        self._plan_mode: bool = self._runtime.session.state.plan_mode
        self._plan_session_id: str | None = self._runtime.session.state.plan_session_id
        # Pre-warm slug cache so the persisted slug survives process restarts
        if self._plan_session_id is not None and self._runtime.session.state.plan_slug is not None:
            from kimi_cli.tools.plan.heroes import seed_slug_cache

            seed_slug_cache(self._plan_session_id, self._runtime.session.state.plan_slug)
        self._pending_plan_activation_injection: bool = False
        if self._plan_mode:
            self._ensure_plan_session_id()
        self._injection_providers: list[DynamicInjectionProvider] = [
            PlanModeInjectionProvider(),
            YoloModeInjectionProvider(),
        ]
        self._hook_engine: HookEngine = HookEngine()
        self._stop_hook_active: bool = False
        if self._runtime.role == "root":
            self._runtime.notifications.ack_ids("llm", extract_notification_ids(context.history))

        # Bind plan mode state to tools that support it
        self._bind_plan_mode_tools()

        self._slash_commands = self._build_slash_commands()
        self._slash_command_map = self._index_slash_commands(self._slash_commands)

    @property
    def name(self) -> str:
        return self._agent.name

    @property
    def model_name(self) -> str:
        return self._runtime.llm.chat_provider.model_name if self._runtime.llm else ""

    @property
    def model_capabilities(self) -> set[ModelCapability] | None:
        if self._runtime.llm is None:
            return None
        return self._runtime.llm.capabilities

    @property
    def is_yolo(self) -> bool:
        """Whether yolo (auto-approve / non-interactive) mode is enabled."""
        return self._approval.is_yolo()

    @property
    def plan_mode(self) -> bool:
        """Whether plan mode (read-only research and planning) is active."""
        return self._plan_mode

    @property
    def hook_engine(self) -> HookEngine:
        return self._hook_engine

    def set_hook_engine(self, engine: HookEngine) -> None:
        self._hook_engine = engine
        if isinstance(self._agent.toolset, KimiToolset):
            self._agent.toolset.set_hook_engine(engine)

    def add_injection_provider(self, provider: DynamicInjectionProvider) -> None:
        """Register an additional dynamic injection provider."""
        self._injection_providers.append(provider)

    async def _collect_injections(self) -> list[DynamicInjection]:
        """Collect dynamic injections from all registered providers."""
        injections: list[DynamicInjection] = []
        for provider in self._injection_providers:
            try:
                result = await provider.get_injections(self._context.history, self)
                injections.extend(result)
            except Exception:
                logger.warning(
                    "injection provider %s failed",
                    type(provider).__name__,
                    exc_info=True,
                )
        return injections

    def _bind_plan_mode_tools(self) -> None:
        """Bind plan mode state to tools that support it."""
        if not isinstance(self._agent.toolset, KimiToolset):
            return

        def checker() -> bool:
            return self._plan_mode

        def path_getter() -> Path | None:
            return self.get_plan_file_path()

        # WriteFile gets both checker and path_getter (for plan file auto-approve)
        from kimi_cli.tools.file.write import WriteFile

        write_tool = self._agent.toolset.find("WriteFile")
        if isinstance(write_tool, WriteFile):
            write_tool.bind_plan_mode(checker, path_getter)

        from kimi_cli.tools.file.replace import StrReplaceFile

        replace_tool = self._agent.toolset.find("StrReplaceFile")
        if isinstance(replace_tool, StrReplaceFile):
            replace_tool.bind_plan_mode(checker, path_getter)

        # ExitPlanMode has a special bind() method
        from kimi_cli.tools.plan import ExitPlanMode

        exit_tool = self._agent.toolset.find("ExitPlanMode")
        if isinstance(exit_tool, ExitPlanMode):
            exit_tool.bind(self.toggle_plan_mode, path_getter, checker, self._approval.is_yolo)

        # EnterPlanMode has a special bind() method
        from kimi_cli.tools.plan.enter import EnterPlanMode

        enter_tool = self._agent.toolset.find("EnterPlanMode")
        if isinstance(enter_tool, EnterPlanMode):
            enter_tool.bind(self.toggle_plan_mode, path_getter, checker, self._approval.is_yolo)

        # AskUserQuestion — bind yolo checker for auto-dismiss
        from kimi_cli.tools.ask_user import AskUserQuestion

        ask_tool = self._agent.toolset.find("AskUserQuestion")
        if isinstance(ask_tool, AskUserQuestion):
            ask_tool.bind_approval(self._approval.is_yolo)

    def _ensure_plan_session_id(self) -> None:
        """Allocate a stable plan session ID on first activation."""
        if self._plan_session_id is None:
            import uuid

            self._plan_session_id = uuid.uuid4().hex
            self._runtime.session.state.plan_session_id = self._plan_session_id
            # Compute and persist slug immediately so the path survives process restarts
            from kimi_cli.tools.plan.heroes import get_or_create_slug

            slug = get_or_create_slug(self._plan_session_id)
            self._runtime.session.state.plan_slug = slug
            self._runtime.session.save_state()

    def _set_plan_mode(self, enabled: bool, *, source: Literal["manual", "tool"]) -> bool:
        """Update plan mode state for either manual or tool-driven toggles."""
        if enabled == self._plan_mode:
            return self._plan_mode
        self._plan_mode = enabled
        if enabled:
            self._ensure_plan_session_id()
            self._pending_plan_activation_injection = source == "manual"
        else:
            self._pending_plan_activation_injection = False
            self._plan_session_id = None
            self._runtime.session.state.plan_session_id = None
            self._runtime.session.state.plan_slug = None
        # Persist plan mode to session state so it survives process restarts
        self._runtime.session.state.plan_mode = self._plan_mode
        self._runtime.session.save_state()
        return self._plan_mode

    def get_plan_file_path(self) -> Path | None:
        """Get the plan file path for the current session."""
        if self._plan_session_id is None:
            return None
        from kimi_cli.tools.plan.heroes import get_plan_file_path

        return get_plan_file_path(self._plan_session_id)

    def read_current_plan(self) -> str | None:
        """Read the current plan file content."""
        if self._plan_session_id is None:
            return None
        from kimi_cli.tools.plan.heroes import read_plan_file

        return read_plan_file(self._plan_session_id)

    def clear_current_plan(self) -> None:
        """Delete the current plan file."""
        path = self.get_plan_file_path()
        if path and path.exists():
            path.unlink()

    async def toggle_plan_mode(self) -> bool:
        """Toggle plan mode on/off. Returns the new state.

        Tools are not hidden/unhidden — instead, each tool checks plan mode
        state at call time and rejects if blocked.
        Periodic reminders are handled by the dynamic injection system.
        """
        return self._set_plan_mode(not self._plan_mode, source="tool")

    async def toggle_plan_mode_from_manual(self) -> bool:
        """Toggle plan mode from UI/manual entry points (slash command, keybinding)."""
        return self._set_plan_mode(not self._plan_mode, source="manual")

    async def set_plan_mode_from_manual(self, enabled: bool) -> bool:
        """Set plan mode to a specific state from UI/manual entry points.

        Unlike toggle, this accepts the desired state directly, avoiding
        race conditions when the caller already knows the target value.
        """
        return self._set_plan_mode(enabled, source="manual")

    def schedule_plan_activation_reminder(self) -> None:
        """Schedule a plan-mode activation reminder for the next turn.

        Use this when plan mode is already active (e.g. restored session with
        ``--plan`` flag) and ``_set_plan_mode`` would early-return because the
        state hasn't actually changed.
        """
        if self._plan_mode:
            self._pending_plan_activation_injection = True

    def consume_pending_plan_activation_injection(self) -> bool:
        """Consume the next-step activation reminder scheduled by a manual toggle."""
        if not self._plan_mode or not self._pending_plan_activation_injection:
            return False
        self._pending_plan_activation_injection = False
        return True

    @property
    def thinking(self) -> bool | None:
        """Whether thinking mode is enabled."""
        if self._runtime.llm is None:
            return None
        if thinking_effort := self._runtime.llm.chat_provider.thinking_effort:
            return thinking_effort != "off"
        return None

    @property
    def status(self) -> StatusSnapshot:
        token_count = self._context.token_count
        max_size = self._runtime.llm.max_context_size if self._runtime.llm is not None else 0
        return StatusSnapshot(
            context_usage=self._context_usage,
            yolo_enabled=self._approval.is_yolo(),
            plan_mode=self._plan_mode,
            context_tokens=token_count,
            max_context_tokens=max_size,
            mcp_status=self._mcp_status_snapshot(),
        )

    @property
    def agent(self) -> Agent:
        return self._agent

    @property
    def runtime(self) -> Runtime:
        return self._runtime

    @property
    def context(self) -> Context:
        return self._context

    @property
    def _context_usage(self) -> float:
        if self._runtime.llm is not None:
            return self._context.token_count / self._runtime.llm.max_context_size
        return 0.0

    @property
    def wire_file(self) -> WireFile:
        return self._runtime.session.wire_file

    def _mcp_status_snapshot(self):
        if not isinstance(self._agent.toolset, KimiToolset):
            return None
        return self._agent.toolset.mcp_status_snapshot()

    async def start_background_mcp_loading(self) -> bool:
        """Start deferred MCP loading, if any, without exposing toolset internals."""
        if not isinstance(self._agent.toolset, KimiToolset):
            return False
        return await self._agent.toolset.start_deferred_mcp_tool_loading()

    async def wait_for_background_mcp_loading(self) -> None:
        """Wait for any in-flight MCP startup to finish."""
        if not isinstance(self._agent.toolset, KimiToolset):
            return
        await self._agent.toolset.wait_for_mcp_tools()

    async def _checkpoint(self):
        await self._context.checkpoint(self._checkpoint_with_user_message)

    def steer(self, content: str | list[ContentPart]) -> None:
        """Queue a steer message for injection into the current turn."""
        self._steer_queue.put_nowait(content)

    async def _consume_pending_steers(self) -> bool:
        """Drain the steer queue and inject as follow-up user messages.

        Returns True if any steers were consumed.

        Note: /btw is intercepted at the UI layer (``classify_input``) before
        reaching the steer queue, so it never appears here.
        """
        consumed = False
        while not self._steer_queue.empty():
            content = self._steer_queue.get_nowait()
            await self._inject_steer(content)
            wire_send(SteerInput(user_input=content))
            consumed = True
        return consumed

    async def _inject_steer(self, content: str | list[ContentPart]) -> None:
        """Inject a single steer as a regular follow-up user message."""
        parts = cast(
            list[ContentPart],
            [TextPart(text=content)] if isinstance(content, str) else list(content),
        )
        message = Message(role="user", content=parts)
        if self._runtime.llm is None:
            raise LLMNotSet()
        if missing_caps := check_message(message, self._runtime.llm.capabilities):
            raise LLMNotSupported(self._runtime.llm, list(missing_caps))
        await self._context.append_message(message)

    @property
    def available_slash_commands(self) -> list[SlashCommand[Any]]:
        return self._slash_commands

    async def run(self, user_input: str | list[ContentPart]):
        approval_source_token = None
        turn_started = False
        turn_finished = False
        if get_current_approval_source_or_none() is None:
            approval_source_token = set_current_approval_source(
                ApprovalSource(kind="foreground_turn", id=uuid.uuid4().hex)
            )
        try:
            # Refresh OAuth tokens on each turn to avoid idle-time expirations.
            await self._runtime.oauth.ensure_fresh(self._runtime)

            # Set session_id ContextVar for toolset hooks
            from kimi_cli.soul.toolset import set_session_id

            set_session_id(self._runtime.session.id)

            # --- UserPromptSubmit hook ---
            text_input_for_hook = user_input if isinstance(user_input, str) else ""
            from kimi_cli.hooks import events

            hook_results = await self._hook_engine.trigger(
                "UserPromptSubmit",
                matcher_value=text_input_for_hook,
                input_data=events.user_prompt_submit(
                    session_id=self._runtime.session.id,
                    cwd=str(Path.cwd()),
                    prompt=text_input_for_hook,
                ),
            )
            for result in hook_results:
                if result.action == "block":
                    wire_send(TurnBegin(user_input=user_input))
                    turn_started = True
                    wire_send(TextPart(text=result.reason or "Prompt blocked by hook."))
                    wire_send(TurnEnd())
                    turn_finished = True
                    return

            wire_send(TurnBegin(user_input=user_input))
            turn_started = True
            user_message = Message(role="user", content=user_input)
            text_input = user_message.extract_text(" ").strip()

            if command_call := parse_slash_command_call(text_input):
                command = self._find_slash_command(command_call.name)
                if command is None:
                    # this should not happen actually, the shell should have filtered it out
                    wire_send(TextPart(text=f'Unknown slash command "/{command_call.name}".'))
                else:
                    ret = command.func(self, command_call.args)
                    if isinstance(ret, Awaitable):
                        await ret
            elif self._loop_control.max_ralph_iterations != 0:
                runner = FlowRunner.ralph_loop(
                    user_message,
                    self._loop_control.max_ralph_iterations,
                )
                await runner.run(self, "")
            else:
                outcome = await self._turn(user_message)

                # Turn-end question detection: if enabled and the turn produced a
                # final message, check whether it asks the user to pick between options.
                if outcome is not None and self._loop_control.turn_end_question_detection:  # type: ignore[reportUnnecessaryComparison]
                    answer = await self._maybe_ask_turn_end_question(outcome)
                    if answer:
                        wire_send(FollowUpInput(text=answer))
                        await self._turn(Message(role="user", content=answer))

            # --- Stop hook (max 1 re-trigger to prevent infinite loop) ---
            if not self._stop_hook_active:
                stop_results = await self._hook_engine.trigger(
                    "Stop",
                    input_data=events.stop(
                        session_id=self._runtime.session.id,
                        cwd=str(Path.cwd()),
                        stop_hook_active=False,
                    ),
                )
                for result in stop_results:
                    if result.action == "block" and result.reason:
                        self._stop_hook_active = True
                        try:
                            await self._turn(Message(role="user", content=result.reason))
                        finally:
                            self._stop_hook_active = False
                        break

            wire_send(TurnEnd())
            turn_finished = True

            # Auto-set title after first real turn (skip slash commands)
            if not command_call:
                session = self._runtime.session
                if session.state.custom_title is None:
                    from kimi_cli.utils.string import shorten

                    title = shorten(
                        Message(role="user", content=user_input).extract_text(" "),
                        width=50,
                    )
                    if title:
                        from kimi_cli.session_state import (
                            load_session_state,
                            save_session_state,
                        )

                        # Read-modify-write: load fresh state to avoid
                        # overwriting concurrent web changes
                        fresh = load_session_state(session.dir)
                        if fresh.custom_title is None:
                            fresh.custom_title = title
                            save_session_state(fresh, session.dir)
                        session.state.custom_title = fresh.custom_title
        finally:
            if turn_started and not turn_finished:
                wire_send(TurnEnd())
            if approval_source_token is not None:
                reset_current_approval_source(approval_source_token)

    async def _turn(self, user_message: Message) -> TurnOutcome:
        if self._runtime.llm is None:
            raise LLMNotSet()

        if missing_caps := check_message(user_message, self._runtime.llm.capabilities):
            raise LLMNotSupported(self._runtime.llm, list(missing_caps))

        await self._checkpoint()  # this creates the checkpoint 0 on first run
        await self._context.append_message(user_message)
        logger.debug("Appended user message to context")
        return await self._agent_loop()

    def _build_slash_commands(self) -> list[SlashCommand[Any]]:
        commands: list[SlashCommand[Any]] = list(soul_slash_registry.list_commands())
        seen_names = {cmd.name for cmd in commands}

        for skill in self._runtime.skills.values():
            if skill.type not in ("standard", "flow"):
                continue
            name = f"{SKILL_COMMAND_PREFIX}{skill.name}"
            if name in seen_names:
                logger.warning(
                    "Skipping skill slash command /{name}: name already registered",
                    name=name,
                )
                continue
            commands.append(
                SlashCommand(
                    name=name,
                    func=self._make_skill_runner(skill),
                    description=skill.description or "",
                    aliases=[],
                )
            )
            seen_names.add(name)

        for skill in self._runtime.skills.values():
            if skill.type != "flow":
                continue
            if skill.flow is None:
                logger.warning("Flow skill {name} has no flow; skipping", name=skill.name)
                continue
            command_name = f"{FLOW_COMMAND_PREFIX}{skill.name}"
            if command_name in seen_names:
                logger.warning(
                    "Skipping prompt flow slash command /{name}: name already registered",
                    name=command_name,
                )
                continue
            runner = FlowRunner(skill.flow, name=skill.name)
            commands.append(
                SlashCommand(
                    name=command_name,
                    func=runner.run,
                    description=skill.description or "",
                    aliases=[],
                )
            )
            seen_names.add(command_name)

        return commands

    @staticmethod
    def _index_slash_commands(
        commands: list[SlashCommand[Any]],
    ) -> dict[str, SlashCommand[Any]]:
        indexed: dict[str, SlashCommand[Any]] = {}
        for command in commands:
            indexed[command.name] = command
            for alias in command.aliases:
                indexed[alias] = command
        return indexed

    def _find_slash_command(self, name: str) -> SlashCommand[Any] | None:
        return self._slash_command_map.get(name)

    def _make_skill_runner(self, skill: Skill) -> Callable[[KimiSoul, str], None | Awaitable[None]]:
        async def _run_skill(soul: KimiSoul, args: str, *, _skill: Skill = skill) -> None:
            skill_text = await read_skill_text(_skill)
            if skill_text is None:
                wire_send(
                    TextPart(text=f'Failed to load skill "/{SKILL_COMMAND_PREFIX}{_skill.name}".')
                )
                return
            extra = args.strip()
            if extra:
                skill_text = f"{skill_text}\n\nUser request:\n{extra}"
            await soul._turn(Message(role="user", content=skill_text))

        _run_skill.__doc__ = skill.description
        return _run_skill

    async def _agent_loop(self) -> TurnOutcome:
        """The main agent loop for one run."""
        assert self._runtime.llm is not None

        # Discard any stale steers from a previous turn.
        while not self._steer_queue.empty():
            self._steer_queue.get_nowait()

        if isinstance(self._agent.toolset, KimiToolset):
            await self.start_background_mcp_loading()
            loading = bool((snapshot := self._mcp_status_snapshot()) and snapshot.loading)
            if loading:
                wire_send(StatusUpdate(mcp_status=snapshot))
                wire_send(MCPLoadingBegin())
            try:
                await self.wait_for_background_mcp_loading()
            finally:
                if loading:
                    wire_send(StatusUpdate(mcp_status=self._mcp_status_snapshot()))
                    wire_send(MCPLoadingEnd())

        step_no = 0
        while True:
            step_no += 1
            if step_no > self._loop_control.max_steps_per_turn:
                raise MaxStepsReached(self._loop_control.max_steps_per_turn)

            wire_send(StepBegin(n=step_no))
            back_to_the_future: BackToTheFuture | None = None
            step_outcome: StepOutcome | None = None
            try:
                # compact the context if needed
                if should_auto_compact(
                    self._context.token_count_with_pending,
                    self._runtime.llm.max_context_size,
                    trigger_ratio=self._loop_control.compaction_trigger_ratio,
                    reserved_context_size=self._loop_control.reserved_context_size,
                ):
                    logger.info("Context too long, compacting...")
                    try:
                        await self.compact_context()
                    except Exception as compact_err:
                        logger.error(
                            "Context compaction failed at step {step_no}: {error_type}: {error}",
                            step_no=step_no,
                            error_type=type(compact_err).__name__,
                            error=compact_err,
                        )
                        raise

                logger.debug("Beginning step {step_no}", step_no=step_no)
                await self._checkpoint()
                self._denwa_renji.set_n_checkpoints(self._context.n_checkpoints)
                step_outcome = await self._step()
            except BackToTheFuture as e:
                back_to_the_future = e
            except Exception as e:
                # any other exception should interrupt the step
                req_id = getattr(e, "request_id", None)
                logger.error(
                    "Agent step {step_no} failed: {error_type}: {error}"
                    + (" (request_id={request_id})" if req_id else ""),
                    step_no=step_no,
                    error_type=type(e).__name__,
                    error=e,
                    request_id=req_id,
                )
                wire_send(StepInterrupted())
                # --- StopFailure hook ---
                from kimi_cli.hooks import events as _hook_events

                _hook_task = asyncio.create_task(
                    self._hook_engine.trigger(
                        "StopFailure",
                        matcher_value=type(e).__name__,
                        input_data=_hook_events.stop_failure(
                            session_id=self._runtime.session.id,
                            cwd=str(Path.cwd()),
                            error_type=type(e).__name__,
                            error_message=str(e),
                        ),
                    )
                )
                _hook_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                # break the agent loop
                raise

            if step_outcome is not None:
                has_steers = await self._consume_pending_steers()
                if has_steers:
                    continue  # steers injected, force another LLM step
                final_message = (
                    step_outcome.assistant_message
                    if step_outcome.stop_reason == "no_tool_calls"
                    else None
                )
                return TurnOutcome(
                    stop_reason=step_outcome.stop_reason,
                    final_message=final_message,
                    step_count=step_no,
                )

            if back_to_the_future is not None:
                await self._context.revert_to(back_to_the_future.checkpoint_id)
                await self._checkpoint()
                await self._context.append_message(back_to_the_future.messages)

            # Consume any pending steers between steps
            await self._consume_pending_steers()

    async def _step(self) -> StepOutcome | None:
        """Run a single step and return a stop outcome, or None to continue."""
        # already checked in `run`
        assert self._runtime.llm is not None
        chat_provider = self._runtime.llm.chat_provider

        if self._runtime.role == "root":

            async def _append_notification(view: NotificationView) -> None:
                await self._context.append_message(build_notification_message(view, self._runtime))
                # --- Notification hook ---
                from kimi_cli.hooks import events

                _hook_task = asyncio.create_task(
                    self._hook_engine.trigger(
                        "Notification",
                        matcher_value=view.event.type,
                        input_data=events.notification(
                            session_id=self._runtime.session.id,
                            cwd=str(Path.cwd()),
                            sink="llm",
                            notification_type=view.event.type,
                            title=view.event.title,
                            body=view.event.body,
                            severity=view.event.severity,
                        ),
                    )
                )
                _hook_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

            await self._runtime.notifications.deliver_pending(
                "llm",
                limit=4,
                before_claim=self._runtime.background_tasks.reconcile,
                on_notification=_append_notification,
            )

        # Dynamic injection
        injections = await self._collect_injections()
        if injections:
            combined_reminders = "\n".join(system_reminder(inj.content).text for inj in injections)
            await self._context.append_message(
                Message(
                    role="user",
                    content=[TextPart(text=combined_reminders)],
                )
            )

        # Normalize: merge adjacent user messages for clean API input
        effective_history = normalize_history(self._context.history)

        async def _run_step_once() -> StepResult:
            # run an LLM step (may be interrupted)
            return await kosong.step(
                chat_provider,
                self._agent.system_prompt,
                self._agent.toolset,
                effective_history,
                on_message_part=wire_send,
                on_tool_result=wire_send,
            )

        @tenacity.retry(
            retry=retry_if_exception(self._is_retryable_error),
            before_sleep=partial(self._retry_log, "step"),
            wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),
            stop=stop_after_attempt(self._loop_control.max_retries_per_step),
            reraise=True,
        )
        async def _kosong_step_with_retry() -> StepResult:
            return await self._run_with_connection_recovery(
                "step",
                _run_step_once,
                chat_provider=chat_provider,
            )

        t0 = time.monotonic()
        result = await _kosong_step_with_retry()
        llm_elapsed = time.monotonic() - t0
        usage = result.usage
        logger.info(
            "LLM step completed in {elapsed:.1f}s (input={input_tokens}, output={output_tokens})",
            elapsed=llm_elapsed,
            input_tokens=usage.input if usage else "?",
            output_tokens=usage.output if usage else "?",
        )
        status_update = StatusUpdate(
            token_usage=usage, message_id=result.id, plan_mode=self._plan_mode
        )
        if usage is not None:
            # mark the token count for the context before the step
            await self._context.update_token_count(usage.input)
            snap = self.status
            status_update.context_usage = snap.context_usage
            status_update.context_tokens = snap.context_tokens
            status_update.max_context_tokens = snap.max_context_tokens
        wire_send(status_update)

        # wait for all tool results (may be interrupted)
        plan_mode_before_tools = self._plan_mode
        results = await result.tool_results()
        logger.debug("Got tool results: {results}", results=results)

        # If a tool (EnterPlanMode/ExitPlanMode) changed plan mode during execution,
        # send a corrected StatusUpdate so the client sees the up-to-date state.
        if self._plan_mode != plan_mode_before_tools:
            wire_send(StatusUpdate(plan_mode=self._plan_mode))

        # shield the context manipulation from interruption
        await asyncio.shield(self._grow_context(result, results))

        rejected_errors = [
            result.return_value
            for result in results
            if isinstance(result.return_value, ToolRejectedError)
        ]
        if (
            rejected_errors
            and not any(e.has_feedback for e in rejected_errors)
            and self._runtime.role != "subagent"
        ):
            # Pure rejection (no user feedback) — stop the turn.
            # Subagents skip this so the LLM can see the rejection and try
            # an alternative approach instead of terminating immediately.
            _ = self._denwa_renji.fetch_pending_dmail()
            return StepOutcome(stop_reason="tool_rejected", assistant_message=result.message)

        # handle pending D-Mail
        if dmail := self._denwa_renji.fetch_pending_dmail():
            assert dmail.checkpoint_id >= 0, "DenwaRenji guarantees checkpoint_id >= 0"
            assert dmail.checkpoint_id < self._context.n_checkpoints, (
                "DenwaRenji guarantees checkpoint_id < n_checkpoints"
            )
            # raise to let the main loop take us back to the future
            raise BackToTheFuture(
                dmail.checkpoint_id,
                [
                    Message(
                        role="user",
                        content=[
                            system(
                                "You just got a D-Mail from your future self. "
                                "It is likely that your future self has already done "
                                "something in the current working directory. Please read "
                                "the D-Mail and decide what to do next. You MUST NEVER "
                                "mention to the user about this information. "
                                f"D-Mail content:\n\n{dmail.message.strip()}"
                            )
                        ],
                    )
                ],
            )

        if result.tool_calls:
            return None
        return StepOutcome(stop_reason="no_tool_calls", assistant_message=result.message)

    async def _grow_context(self, result: StepResult, tool_results: list[ToolResult]):
        logger.debug("Growing context with result: {result}", result=result)

        assert self._runtime.llm is not None
        tool_messages = [tool_result_to_message(tr) for tr in tool_results]
        for tm in tool_messages:
            if missing_caps := check_message(tm, self._runtime.llm.capabilities):
                logger.warning(
                    "Tool result message requires unsupported capabilities: {caps}",
                    caps=missing_caps,
                )
                raise LLMNotSupported(self._runtime.llm, list(missing_caps))

        await self._context.append_message(result.message)
        if result.usage is not None:
            await self._context.update_token_count(result.usage.total)

        logger.debug(
            "Appending tool messages to context: {tool_messages}", tool_messages=tool_messages
        )
        await self._context.append_message(tool_messages)
        # token count of tool results are not available yet

    async def compact_context(self, custom_instruction: str = "") -> None:
        """
        Compact the context.

        Raises:
            LLMNotSet: When the LLM is not set.
            ChatProviderError: When the chat provider returns an error.
        """

        chat_provider = self._runtime.llm.chat_provider if self._runtime.llm is not None else None

        async def _run_compaction_once() -> CompactionResult:
            if self._runtime.llm is None:
                raise LLMNotSet()
            return await self._compaction.compact(
                self._context.history, self._runtime.llm, custom_instruction=custom_instruction
            )

        @tenacity.retry(
            retry=retry_if_exception(self._is_retryable_error),
            before_sleep=partial(self._retry_log, "compaction"),
            wait=wait_exponential_jitter(initial=0.3, max=5, jitter=0.5),
            stop=stop_after_attempt(self._loop_control.max_retries_per_step),
            reraise=True,
        )
        async def _compact_with_retry() -> CompactionResult:
            return await self._run_with_connection_recovery(
                "compaction",
                _run_compaction_once,
                chat_provider=chat_provider,
            )

        trigger_reason = "manual" if custom_instruction else "auto"
        from kimi_cli.hooks import events

        await self._hook_engine.trigger(
            "PreCompact",
            matcher_value=trigger_reason,
            input_data=events.pre_compact(
                session_id=self._runtime.session.id,
                cwd=str(Path.cwd()),
                trigger=trigger_reason,
                token_count=self._context.token_count,
            ),
        )

        wire_send(CompactionBegin())
        compaction_result = await _compact_with_retry()
        await self._context.clear()
        await self._context.write_system_prompt(self._agent.system_prompt)
        await self._checkpoint()
        await self._context.append_message(compaction_result.messages)
        estimated_token_count = compaction_result.estimated_token_count

        if self._runtime.role == "root":
            active_task_snapshot = build_active_task_snapshot(self._runtime.background_tasks)
            if active_task_snapshot is not None:
                active_task_message = Message(
                    role="user",
                    content=[
                        system(
                            "The following background tasks are still active after compaction. "
                            "Use TaskList if you need to re-enumerate them later."
                        ),
                        TextPart(text=active_task_snapshot),
                    ],
                )
                await self._context.append_message(active_task_message)
                estimated_token_count += estimate_text_tokens([active_task_message])

        # Estimate token count so context_usage is not reported as 0%
        await self._context.update_token_count(estimated_token_count)

        wire_send(CompactionEnd())

        _hook_task = asyncio.create_task(
            self._hook_engine.trigger(
                "PostCompact",
                matcher_value=trigger_reason,
                input_data=events.post_compact(
                    session_id=self._runtime.session.id,
                    cwd=str(Path.cwd()),
                    trigger=trigger_reason,
                    estimated_token_count=estimated_token_count,
                ),
            )
        )
        _hook_task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    @staticmethod
    def _is_retryable_error(exception: BaseException) -> bool:
        if isinstance(exception, (APIConnectionError, APITimeoutError)):
            return not bool(getattr(exception, "_kimi_recovery_exhausted", False))
        if isinstance(exception, APIEmptyResponseError):
            return True
        return isinstance(exception, APIStatusError) and exception.status_code in (
            429,  # Too Many Requests
            500,  # Internal Server Error
            502,  # Bad Gateway
            503,  # Service Unavailable
            504,  # Gateway Timeout
        )

    async def _run_with_connection_recovery(
        self,
        name: str,
        operation: Callable[[], Awaitable[Any]],
        *,
        chat_provider: object | None = None,
        _auth_retried: bool = False,
    ) -> Any:
        try:
            return await operation()
        except APIStatusError as error:
            if error.status_code != 401 or _auth_retried:
                raise
            # Only attempt refresh+retry when the active model's provider
            # uses OAuth.  For plain API-key providers there is nothing
            # to refresh and retrying would just add latency.
            active_provider = (
                self._runtime.config.providers.get(self._runtime.llm.model_config.provider)
                if self._runtime.llm and self._runtime.llm.model_config
                else None
            )
            if not (active_provider and active_provider.oauth):
                raise
            logger.warning(
                "Received 401 during {name}, attempting token refresh",
                name=name,
            )
            try:
                await self._runtime.oauth.ensure_fresh(self._runtime, force=True)
            except Exception as refresh_exc:
                logger.exception("Token refresh failed after 401.")
                raise error from refresh_exc
            # Re-enter full recovery so that transient connection errors
            # on the retry are still handled by on_retryable_error.
            return await self._run_with_connection_recovery(
                name, operation, chat_provider=chat_provider, _auth_retried=True
            )
        except (APIConnectionError, APITimeoutError) as error:
            if not isinstance(chat_provider, RetryableChatProvider):
                raise
            try:
                recovered = chat_provider.on_retryable_error(error)
            except Exception:
                logger.exception(
                    "Failed to recover chat provider during {name} after {error_type}.",
                    name=name,
                    error_type=type(error).__name__,
                )
                raise
            if not recovered:
                logger.warning(
                    "Chat provider recovery not available for {name} after {error_type}.",
                    name=name,
                    error_type=type(error).__name__,
                )
                raise
            logger.info(
                "Recovered chat provider during {name} after {error_type}; retrying once.",
                name=name,
                error_type=type(error).__name__,
            )
            try:
                return await operation()
            except (APIConnectionError, APITimeoutError) as second_error:
                logger.warning(
                    "Chat provider recovery exhausted for {name}: {error_type}: {error}",
                    name=name,
                    error_type=type(second_error).__name__,
                    error=second_error,
                )
                second_error._kimi_recovery_exhausted = True  # type: ignore[attr-defined]
                raise

    @staticmethod
    def _retry_log(name: str, retry_state: RetryCallState):
        error = retry_state.outcome.exception() if retry_state.outcome else None
        logger.warning(
            "Retrying {name} for the {n} time (last error: {error_type}: {error}). "
            "Waiting {sleep} seconds.",
            name=name,
            n=retry_state.attempt_number,
            error_type=type(error).__name__ if error else "unknown",
            error=error or "unknown",
            sleep=retry_state.next_action.sleep
            if retry_state.next_action is not None
            else "unknown",
        )

    @staticmethod
    def _extract_json_payload(text: str) -> str:
        payload = text.strip()
        if payload.startswith("```"):
            lines = payload.splitlines()
            if len(lines) >= 3 and lines[-1].strip().startswith("```"):
                payload = "\n".join(lines[1:-1]).strip()
        start = payload.find("{")
        end = payload.rfind("}")
        if start >= 0 and end > start:
            return payload[start : end + 1]
        return payload

    # -- Turn-end question detection ---------------------------------------------------

    _TURN_END_DETECT_MAX_ATTEMPTS = 2
    _TURN_END_DETECT_TIMEOUT = 15.0  # seconds
    _TURN_END_DETECT_CONTEXT_TURNS = 3
    _TURN_END_DETECT_CONTEXT_CHARS = 600
    _TURN_END_DETECT_EXCERPT_CHARS = 600
    _TURN_END_DETECT_LATEST_MESSAGE_CHARS = 2000

    async def _detect_turn_end_question(
        self,
        assistant_message: Message,
    ) -> TurnEndQuestionDetection | None:
        """Use a side-channel LLM call to check if *assistant_message* asks the user
        to choose between options.  Returns the parsed detection or ``None`` on
        failure.  Retries up to ``_TURN_END_DETECT_MAX_ATTEMPTS`` when the LLM
        returns unparseable output.  The entire detection is capped at
        ``_TURN_END_DETECT_TIMEOUT`` seconds."""
        try:
            return await asyncio.wait_for(
                self._detect_turn_end_question_inner(assistant_message),
                timeout=self._TURN_END_DETECT_TIMEOUT,
            )
        except TimeoutError:
            logger.warning(
                "Turn-end question detection timed out after {timeout}s",
                timeout=self._TURN_END_DETECT_TIMEOUT,
            )
            return self._heuristic_turn_end_question(assistant_message.extract_text(" "))

    def _turn_end_question_excerpt(self, text: str) -> str:
        units = [
            unit.strip()
            for unit in re.split(r"(?:\r?\n)+|(?<=[\u3002\uff01\uff1f!?])\s*", text)
            if unit.strip()
        ]
        if not units:
            return text.strip()
        return "\n".join(units[-3:])

    def _clip_turn_end_detector_text(self, text: str, *, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        separator = "\n...\n"
        keep = max(1, (max_chars - len(separator)) // 2)
        return f"{text[:keep].rstrip()}{separator}{text[-keep:].lstrip()}"

    def _recent_turn_end_detection_context(
        self,
        assistant_message: Message,
        *,
        max_turns: int,
    ) -> list[tuple[str, str]]:
        history = self._context.history
        messages = reversed(history)
        if not history or history[-1] != assistant_message:
            messages = chain((assistant_message,), messages)

        turns_reversed: list[tuple[str, str]] = []
        current_assistant: str | None = None

        for msg in messages:
            if msg.role == "assistant" and current_assistant is None:
                current_assistant = msg.extract_text(sep="\n").strip()

            if not is_real_user_turn_start_message(msg):
                continue

            turns_reversed.append((msg.extract_text(sep="\n").strip(), current_assistant or ""))
            current_assistant = None
            if len(turns_reversed) >= max_turns:
                break

        turns_reversed.reverse()
        return turns_reversed

    def _build_turn_end_detector_prompt_input(self, assistant_message: Message) -> str:
        text_only = assistant_message.extract_text(" ").strip()
        excerpt = self._clip_turn_end_detector_text(
            self._turn_end_question_excerpt(text_only),
            max_chars=self._TURN_END_DETECT_EXCERPT_CHARS,
        )
        latest_message = self._clip_turn_end_detector_text(
            text_only,
            max_chars=self._TURN_END_DETECT_LATEST_MESSAGE_CHARS,
        )
        recent_turns = self._recent_turn_end_detection_context(
            assistant_message,
            max_turns=self._TURN_END_DETECT_CONTEXT_TURNS,
        )
        if recent_turns and recent_turns[-1][1] == text_only and text_only:
            recent_turns = [
                *recent_turns[:-1],
                (recent_turns[-1][0], "(latest message shown below)"),
            ]

        lines = [
            (
                "Analyze whether the latest assistant message asks the user to choose "
                "between options or make a decision."
            ),
            (
                "Use recent turns only as supporting context. Base has_question on "
                "the latest assistant message, not on older turns."
            ),
        ]
        if recent_turns:
            lines.extend(
                [
                    "",
                    f"Recent turns (last {len(recent_turns)}, oldest to newest):",
                ]
            )
            for idx, (user_text, assistant_text) in enumerate(recent_turns, start=1):
                clipped_user = self._clip_turn_end_detector_text(
                    user_text,
                    max_chars=self._TURN_END_DETECT_CONTEXT_CHARS,
                )
                clipped_assistant = self._clip_turn_end_detector_text(
                    assistant_text,
                    max_chars=self._TURN_END_DETECT_CONTEXT_CHARS,
                )
                lines.extend(
                    [
                        "",
                        f"[Turn {idx}]",
                        f"User:\n{clipped_user or '(empty)'}",
                        f"Assistant:\n{clipped_assistant or '(no textual reply)'}",
                    ]
                )

        lines.extend(
            [
                "",
                (
                    "Focus on the ending of the latest assistant message, but use "
                    "the full latest message if earlier lines contain the options."
                ),
                "",
                "Latest message ending excerpt:",
                excerpt,
                "",
                "Latest full assistant message (trimmed if needed):",
                latest_message,
            ]
        )
        return "\n".join(lines)

    def _heuristic_turn_end_question(
        self,
        assistant_text: str,
    ) -> TurnEndQuestionDetection | None:
        excerpt = self._turn_end_question_excerpt(assistant_text)
        units = [
            unit.strip().strip('\u201c\u201d\u201e\u201f""\'`')  # noqa: B005
            for unit in re.split(r"(?:\r?\n)+|(?<=[\u3002\uff01\uff1f!?])\s*", excerpt)
            if unit.strip()
        ]
        if not units:
            return None

        soft_prefixes = (
            "如果你要",
            "如果你想",
            "如果你愿意",
            "如果你希望",
            "如果继续",
            "如果要继续",
        )
        leading_wrappers = '>》」』】）)]-•·*\u201c\u201d\u201e\u201f""\'`('
        conditional_offer_tokens = ("我可以", "我现在就", "我现在可以", "我现在就可以")
        direct_offer_tokens = conditional_offer_tokens + ("我就",)
        action_tokens = (
            "继续",
            "开始",
            "按这个方案",
            "修改",
            "处理",
            "推进",
            "做下去",
            "改下去",
            "做下一轮",
            "做下一步",
        )

        for unit in reversed(units):
            normalized = re.sub(r"\s+", "", unit)
            if not normalized:
                continue

            candidate = normalized.lstrip(leading_wrappers)
            prefix = next((token for token in soft_prefixes if candidate.startswith(token)), None)
            if prefix is None:
                continue
            if candidate.startswith(
                ("如果继续这样做", "如果继续这么做", "如果要继续这样做", "如果要继续这么做")
            ):
                continue

            offer_tokens = (
                direct_offer_tokens
                if prefix in {"如果你要", "如果你想", "如果你愿意", "如果你希望"}
                else conditional_offer_tokens
            )
            if not any(token in candidate for token in offer_tokens):
                continue
            if not any(token in candidate for token in action_tokens):
                continue

            continue_like = any(
                token in candidate for token in ("继续", "做下去", "改下去", "做下一轮", "做下一步")
            )
            if continue_like:
                question = "要我继续吗？"
                options = (
                    TurnEndQuestionOption(label="继续", description="继续按当前方案往下做"),
                    TurnEndQuestionOption(label="先别", description="先不要继续"),
                )
            else:
                question = "要我现在开始吗？"
                options = (
                    TurnEndQuestionOption(label="开始", description="现在开始处理"),
                    TurnEndQuestionOption(label="先别", description="先不要开始"),
                )
            return TurnEndQuestionDetection(
                has_question=True,
                questions=(TurnEndQuestionItem(question=question, options=options),),
            )
        return None

    async def _detect_turn_end_question_inner(
        self,
        assistant_message: Message,
    ) -> TurnEndQuestionDetection | None:
        """Inner implementation without timeout wrapper."""
        assert self._runtime.llm is not None
        chat_provider = self._runtime.llm.chat_provider.with_thinking("off")

        text_only = assistant_message.extract_text(" ").strip()
        if not text_only:
            return None
        heuristic_detection = self._heuristic_turn_end_question(text_only)
        history: list[Message] = [
            Message(
                role="user",
                content=self._build_turn_end_detector_prompt_input(assistant_message),
            )
        ]

        for attempt in range(1, self._TURN_END_DETECT_MAX_ATTEMPTS + 1):

            async def _run_once():
                return await kosong.generate(
                    chat_provider=chat_provider,
                    system_prompt=TURN_END_QUESTION_DETECTOR_PROMPT,
                    tools=[],
                    history=history,
                )

            try:
                result = await self._run_with_connection_recovery(
                    "turn-end question detection",
                    _run_once,
                    chat_provider=chat_provider,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Turn-end question detection failed: {error}", error=exc)
                return heuristic_detection

            raw_text = result.message.extract_text(" ")
            detection = self._parse_turn_end_question_payload(raw_text)
            if detection is not None:
                return detection

            if attempt < self._TURN_END_DETECT_MAX_ATTEMPTS:
                logger.debug(
                    "Turn-end question detection returned unparseable output "
                    "(attempt {attempt}), retrying",
                    attempt=attempt,
                )
            else:
                logger.warning(
                    "Turn-end question detection returned unparseable output "
                    "after {attempts} attempts; giving up",
                    attempts=self._TURN_END_DETECT_MAX_ATTEMPTS,
                )

        return heuristic_detection

    def _parse_turn_end_question_payload(self, text: str) -> TurnEndQuestionDetection | None:
        payload_text = self._extract_json_payload(text)
        try:
            payload_obj: object = json.loads(payload_text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload_obj, dict):
            return None
        payload = cast(dict[str, object], payload_obj)

        has_question = payload.get("has_question", False)
        if not isinstance(has_question, bool):
            return None
        if not has_question:
            return TurnEndQuestionDetection(has_question=False, questions=())

        raw_questions = payload.get("questions", [])
        if not isinstance(raw_questions, list):
            return None

        questions: list[TurnEndQuestionItem] = []
        for raw_q in cast(list[object], raw_questions)[:4]:
            if not isinstance(raw_q, dict):
                continue
            q = cast(dict[str, object], raw_q)
            question_text = q.get("question")
            if not isinstance(question_text, str) or not question_text.strip():
                continue
            raw_options = q.get("options", [])
            if not isinstance(raw_options, list):
                continue
            options: list[TurnEndQuestionOption] = []
            for raw_opt in cast(list[object], raw_options)[:4]:
                if not isinstance(raw_opt, dict):
                    continue
                opt = cast(dict[str, object], raw_opt)
                label = opt.get("label")
                if not isinstance(label, str) or not label.strip():
                    continue
                desc = opt.get("description", "")
                desc = desc.strip() if isinstance(desc, str) else ""
                options.append(TurnEndQuestionOption(label=label.strip(), description=desc))
            if len(options) >= 2:
                questions.append(
                    TurnEndQuestionItem(
                        question=question_text.strip(),
                        options=tuple(options),
                    )
                )
        if not questions:
            return TurnEndQuestionDetection(has_question=False, questions=())
        return TurnEndQuestionDetection(has_question=True, questions=tuple(questions))

    async def _maybe_ask_turn_end_question(
        self,
        outcome: TurnOutcome,
    ) -> str | None:
        """Detect choice questions in the turn's final message and present them
        to the user via a structured ``QuestionRequest``.

        Returns the user's answer text to be used as the next turn prompt,
        or ``None`` if no question was detected / the user dismissed it.
        """
        if outcome.stop_reason != "no_tool_calls" or outcome.final_message is None:
            return None

        detection = await self._detect_turn_end_question(outcome.final_message)
        if detection is None or not detection.has_question:
            return None

        wire = get_wire_or_none()
        if wire is None:
            return None

        assistant_reply_body = outcome.final_message.extract_text(sep="\n").strip()

        questions = [
            QuestionItem(
                question=q.question,
                options=[
                    QuestionOption(label=o.label, description=o.description) for o in q.options
                ],
                body=assistant_reply_body,
            )
            for q in detection.questions
        ]

        request = QuestionRequest(
            id=str(uuid4()),
            tool_call_id=f"turn-end-{uuid4().hex[:8]}",
            questions=questions,
        )

        wire_send(request)

        try:
            answers = await request.wait()
        except QuestionNotSupported:
            logger.debug("Client does not support interactive questions; skipping")
            return None
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Failed to get user response for turn-end question")
            return None

        if not answers:
            return None

        parts: list[str] = []
        for _question_text, answer_text in answers.items():
            parts.append(f"{answer_text}")
        return "\n".join(parts) if parts else None


class BackToTheFuture(Exception):
    """
    Raise when we need to revert the context to a previous checkpoint.
    The main agent loop should catch this exception and handle it.
    """

    def __init__(self, checkpoint_id: int, messages: Sequence[Message]):
        self.checkpoint_id = checkpoint_id
        self.messages = messages


class FlowRunner:
    def __init__(
        self,
        flow: Flow,
        *,
        name: str | None = None,
        max_moves: int = DEFAULT_MAX_FLOW_MOVES,
    ) -> None:
        self._flow = flow
        self._name = name
        self._max_moves = max_moves

    @staticmethod
    def ralph_loop(
        user_message: Message,
        max_ralph_iterations: int,
    ) -> FlowRunner:
        prompt_content = list(user_message.content)
        prompt_text = Message(role="user", content=prompt_content).extract_text(" ").strip()
        total_runs = max_ralph_iterations + 1
        if max_ralph_iterations < 0:
            total_runs = 1000000000000000  # effectively infinite

        nodes: dict[str, FlowNode] = {
            "BEGIN": FlowNode(id="BEGIN", label="BEGIN", kind="begin"),
            "END": FlowNode(id="END", label="END", kind="end"),
        }
        outgoing: dict[str, list[FlowEdge]] = {"BEGIN": [], "END": []}

        nodes["R1"] = FlowNode(id="R1", label=prompt_content, kind="task")
        nodes["R2"] = FlowNode(
            id="R2",
            label=(
                f"{prompt_text}. (You are running in an automated loop where the same "
                "prompt is fed repeatedly. Only choose STOP when the task is fully complete. "
                "Including it will stop further iterations. If you are not 100% sure, "
                "choose CONTINUE.)"
            ).strip(),
            kind="decision",
        )
        outgoing["R1"] = []
        outgoing["R2"] = []

        outgoing["BEGIN"].append(FlowEdge(src="BEGIN", dst="R1", label=None))
        outgoing["R1"].append(FlowEdge(src="R1", dst="R2", label=None))
        outgoing["R2"].append(FlowEdge(src="R2", dst="R2", label="CONTINUE"))
        outgoing["R2"].append(FlowEdge(src="R2", dst="END", label="STOP"))

        flow = Flow(nodes=nodes, outgoing=outgoing, begin_id="BEGIN", end_id="END")
        max_moves = total_runs
        return FlowRunner(flow, max_moves=max_moves)

    async def run(self, soul: KimiSoul, args: str) -> None:
        if args.strip():
            command = f"/{FLOW_COMMAND_PREFIX}{self._name}" if self._name else "/flow"
            logger.warning("Agent flow {command} ignores args: {args}", command=command, args=args)
            return

        current_id = self._flow.begin_id
        moves = 0
        total_steps = 0
        while True:
            node = self._flow.nodes[current_id]
            edges = self._flow.outgoing.get(current_id, [])

            if node.kind == "end":
                logger.info("Agent flow reached END node {node_id}", node_id=current_id)
                return

            if node.kind == "begin":
                if not edges:
                    logger.error(
                        'Agent flow BEGIN node "{node_id}" has no outgoing edges; stopping.',
                        node_id=node.id,
                    )
                    return
                current_id = edges[0].dst
                continue

            if moves >= self._max_moves:
                raise MaxStepsReached(total_steps)
            next_id, steps_used = await self._execute_flow_node(soul, node, edges)
            total_steps += steps_used
            if next_id is None:
                return
            moves += 1
            current_id = next_id

    async def _execute_flow_node(
        self,
        soul: KimiSoul,
        node: FlowNode,
        edges: list[FlowEdge],
    ) -> tuple[str | None, int]:
        if not edges:
            logger.error(
                'Agent flow node "{node_id}" has no outgoing edges; stopping.',
                node_id=node.id,
            )
            return None, 0

        base_prompt = self._build_flow_prompt(node, edges)
        prompt = base_prompt
        steps_used = 0
        while True:
            result = await self._flow_turn(soul, prompt)
            steps_used += result.step_count
            if result.stop_reason == "tool_rejected":
                logger.error("Agent flow stopped after tool rejection.")
                return None, steps_used

            if node.kind != "decision":
                return edges[0].dst, steps_used

            choice = (
                parse_choice(result.final_message.extract_text(" "))
                if result.final_message
                else None
            )
            next_id = self._match_flow_edge(edges, choice)
            if next_id is not None:
                return next_id, steps_used

            options = ", ".join(edge.label or "" for edge in edges)
            logger.warning(
                "Agent flow invalid choice. Got: {choice}. Available: {options}.",
                choice=choice or "<missing>",
                options=options,
            )
            prompt = (
                f"{base_prompt}\n\n"
                "Your last response did not include a valid choice. "
                "Reply with one of the choices using <choice>...</choice>."
            )

    @staticmethod
    def _build_flow_prompt(node: FlowNode, edges: list[FlowEdge]) -> str | list[ContentPart]:
        if node.kind != "decision":
            return node.label

        if not isinstance(node.label, str):
            label_text = Message(role="user", content=node.label).extract_text(" ")
        else:
            label_text = node.label
        choices = [edge.label for edge in edges if edge.label]
        lines = [
            label_text,
            "",
            "Available branches:",
            *(f"- {choice}" for choice in choices),
            "",
            "Reply with a choice using <choice>...</choice>.",
        ]
        return "\n".join(lines)

    @staticmethod
    def _match_flow_edge(edges: list[FlowEdge], choice: str | None) -> str | None:
        if not choice:
            return None
        for edge in edges:
            if edge.label == choice:
                return edge.dst
        return None

    @staticmethod
    async def _flow_turn(
        soul: KimiSoul,
        prompt: str | list[ContentPart],
    ) -> TurnOutcome:
        wire_send(TurnBegin(user_input=prompt))
        res = await soul._turn(Message(role="user", content=prompt))  # type: ignore[reportPrivateUsage]
        wire_send(TurnEnd())
        return res

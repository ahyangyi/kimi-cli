"""Tests for turn-end question detection."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import pytest
from kosong.message import ImageURLPart, Message
from kosong.message import TextPart as KosongTextPart
from kosong.message import ThinkPart as KosongThinkPart
from kosong.tooling.empty import EmptyToolset

import kimi_cli.soul as soul_module
import kimi_cli.soul.kimisoul as kimisoul_module
from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import (
    KimiSoul,
    TurnEndQuestionDetection,
    TurnEndQuestionItem,
    TurnEndQuestionOption,
    TurnOutcome,
)
from kimi_cli.wire import Wire
from kimi_cli.wire.types import FollowUpInput, QuestionRequest

# -- Payload parsing tests --


class TestParseTurnEndQuestionPayload:
    def _make_soul(self, runtime: Runtime, tmp_path: Path) -> KimiSoul:
        return KimiSoul(
            Agent(
                name="Test",
                system_prompt="Test",
                toolset=EmptyToolset(),
                runtime=runtime,
            ),
            context=Context(file_backend=tmp_path / "history.jsonl"),
        )

    def test_parse_no_question(self, runtime: Runtime, tmp_path: Path) -> None:
        soul = self._make_soul(runtime, tmp_path)
        result = soul._parse_turn_end_question_payload('{"has_question": false, "questions": []}')
        assert result == TurnEndQuestionDetection(has_question=False, questions=())

    def test_parse_with_question(self, runtime: Runtime, tmp_path: Path) -> None:
        soul = self._make_soul(runtime, tmp_path)
        result = soul._parse_turn_end_question_payload(
            '{"has_question": true, "questions": [{"question": "Pick A or B?",'
            ' "options": [{"label": "A", "description": "Option A"},'
            ' {"label": "B", "description": "Option B"}]}]}'
        )
        assert result == TurnEndQuestionDetection(
            has_question=True,
            questions=(
                TurnEndQuestionItem(
                    question="Pick A or B?",
                    options=(
                        TurnEndQuestionOption(label="A", description="Option A"),
                        TurnEndQuestionOption(label="B", description="Option B"),
                    ),
                ),
            ),
        )

    def test_parse_malformed_json(self, runtime: Runtime, tmp_path: Path) -> None:
        soul = self._make_soul(runtime, tmp_path)
        assert soul._parse_turn_end_question_payload("not json") is None

    def test_parse_question_with_too_few_options_ignored(
        self, runtime: Runtime, tmp_path: Path
    ) -> None:
        soul = self._make_soul(runtime, tmp_path)
        result = soul._parse_turn_end_question_payload(
            '{"has_question": true, "questions": [{"question": "Pick?",'
            ' "options": [{"label": "Only one"}]}]}'
        )
        # question with < 2 options is dropped → no questions → has_question=false
        assert result is not None
        assert result.has_question is False

    def test_parse_json_wrapped_in_markdown(self, runtime: Runtime, tmp_path: Path) -> None:
        soul = self._make_soul(runtime, tmp_path)
        text = (
            "```json\n"
            '{"has_question": true, "questions": [{"question": "A or B?",'
            ' "options": [{"label": "A"}, {"label": "B"}]}]}\n'
            "```"
        )
        result = soul._parse_turn_end_question_payload(text)
        assert result is not None
        assert result.has_question is True
        assert len(result.questions) == 1


# -- Detection LLM call test --


@pytest.mark.asyncio
async def test_detect_turn_end_question_calls_generate(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    captured_system_prompt: str | None = None
    captured_history: list[Message] = []

    async def fake_generate(*, chat_provider, system_prompt, tools, history: Sequence[Message]):
        nonlocal captured_system_prompt, captured_history
        captured_system_prompt = system_prompt
        captured_history = list(history)
        return SimpleNamespace(
            message=Message(
                role="assistant",
                content=(
                    '{"has_question": true, "questions": [{"question": "A or B?",'
                    ' "options": [{"label": "A"}, {"label": "B"}]}]}'
                ),
            )
        )

    monkeypatch.setattr(kimisoul_module.kosong, "generate", fake_generate)

    assistant_msg = Message(role="assistant", content="Should I do A or B?")
    result = await soul._detect_turn_end_question(assistant_msg)

    assert result is not None
    assert result.has_question is True
    assert len(result.questions) == 1
    assert result.questions[0].question == "A or B?"
    # The side-channel should wrap text in a user message for the detector
    assert captured_system_prompt is not None
    assert len(captured_history) == 1
    first_message = captured_history[0]
    assert first_message.role == "user"
    history_text = first_message.extract_text()
    assert "Latest message ending excerpt:" in history_text
    assert "Latest full assistant message (trimmed if needed):" in history_text
    assert "Should I do A or B?" in history_text


@pytest.mark.asyncio
async def test_detect_turn_end_question_includes_recent_three_turns_of_context(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    await soul._context.append_message(
        [
            Message(role="user", content="u1"),
            Message(role="assistant", content="a1"),
            Message(role="user", content="u2"),
            Message(role="assistant", content="a2"),
            Message(role="user", content="u3"),
            Message(role="assistant", content="a3"),
            Message(role="user", content="u4"),
        ]
    )

    captured_history: list[Message] = []

    async def fake_generate(*, chat_provider, system_prompt, tools, history: Sequence[Message]):
        nonlocal captured_history
        captured_history = list(history)
        return SimpleNamespace(
            message=Message(
                role="assistant",
                content='{"has_question": false, "questions": []}',
            )
        )

    monkeypatch.setattr(kimisoul_module.kosong, "generate", fake_generate)

    assistant_msg = Message(role="assistant", content="Should I continue?")
    await soul._detect_turn_end_question(assistant_msg)

    assert len(captured_history) == 1
    history_text = captured_history[0].extract_text()
    assert "Recent turns (last 3, oldest to newest):" in history_text
    assert "User:\nu2" in history_text
    assert "Assistant:\na2" in history_text
    assert "User:\nu3" in history_text
    assert "Assistant:\na3" in history_text
    assert "User:\nu4" in history_text
    assert "Assistant:\n(latest message shown below)" in history_text
    assert history_text.count("Should I continue?") == 2
    assert "User:\nu1" not in history_text
    assert "Assistant:\na1" not in history_text


@pytest.mark.asyncio
async def test_detect_turn_end_question_trims_recent_context_and_latest_message(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    long_user = "user-" + ("u" * 4000)
    long_assistant = "assistant-" + ("a" * 4000)
    latest = "latest-" + ("l" * 4000) + "-done?"
    await soul._context.append_message(
        [
            Message(role="user", content=long_user),
            Message(role="assistant", content=long_assistant),
            Message(role="user", content="follow-up"),
        ]
    )

    prompt = soul._build_turn_end_detector_prompt_input(Message(role="assistant", content=latest))

    assert len(prompt) < 5000
    assert "Assistant:\n(latest message shown below)" in prompt
    assert latest not in prompt
    assert "Latest full assistant message (trimmed if needed):" in prompt


@pytest.mark.asyncio
async def test_recent_turn_end_detection_context_does_not_reuse_old_text_for_non_text_reply(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    assistant_msg = Message(
        role="assistant",
        content=[ImageURLPart(image_url=ImageURLPart.ImageURL(url="https://example.com/x.png"))],
    )
    await soul._context.append_message(
        [
            Message(role="user", content="show me"),
            Message(role="assistant", content="working on it"),
        ]
    )

    turns = soul._recent_turn_end_detection_context(assistant_msg, max_turns=1)

    assert turns == [("show me", "")]


# -- Integration: _maybe_ask_turn_end_question --


@pytest.mark.asyncio
async def test_detect_turn_end_question_strips_thinking_parts(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Thinking/reasoning parts should be stripped before sending to the
    side-channel LLM; only text content should be included."""
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    captured_history: list[Message] = []

    async def fake_generate(*, chat_provider, system_prompt, tools, history: Sequence[Message]):
        nonlocal captured_history
        captured_history = list(history)
        return SimpleNamespace(
            message=Message(
                role="assistant",
                content='{"has_question": false, "questions": []}',
            )
        )

    monkeypatch.setattr(kimisoul_module.kosong, "generate", fake_generate)

    # Build a message with both ThinkPart and TextPart
    assistant_msg = Message(
        role="assistant",
        content=[
            KosongThinkPart(think="Let me reason about this..."),
            KosongTextPart(text="Should I do A or B?"),
        ],
    )
    await soul._detect_turn_end_question(assistant_msg)

    # The side-channel should only see the text content, not the thinking,
    # wrapped in a user message for the detector.
    assert len(captured_history) == 1
    first_message = captured_history[0]
    assert first_message.role == "user"
    assert "Should I do A or B?" in first_message.extract_text()
    # Ensure no ThinkPart in the sent message
    for part in first_message.content:
        assert not isinstance(part, KosongThinkPart)


@pytest.mark.asyncio
async def test_maybe_ask_turn_end_question_no_question(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When detection says no question, _maybe_ask_turn_end_question returns None."""
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    async def fake_detect(self, msg):
        return TurnEndQuestionDetection(has_question=False, questions=())

    monkeypatch.setattr(KimiSoul, "_detect_turn_end_question", fake_detect)

    outcome = TurnOutcome(
        stop_reason="no_tool_calls",
        final_message=Message(role="assistant", content="All done."),
        step_count=1,
    )
    result = await soul._maybe_ask_turn_end_question(outcome)
    assert result is None


@pytest.mark.asyncio
async def test_maybe_ask_turn_end_question_tool_rejected(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the turn stopped due to tool rejection, skip detection."""
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    detect_called = False

    async def fake_detect(self, msg):
        nonlocal detect_called
        detect_called = True
        return TurnEndQuestionDetection(has_question=False, questions=())

    monkeypatch.setattr(KimiSoul, "_detect_turn_end_question", fake_detect)

    outcome = TurnOutcome(
        stop_reason="tool_rejected",
        final_message=None,
        step_count=1,
    )
    result = await soul._maybe_ask_turn_end_question(outcome)
    assert result is None
    assert not detect_called


@pytest.mark.asyncio
async def test_maybe_ask_turn_end_question_sends_question_request(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a question is detected, a QuestionRequest is sent via wire
    and user answer is returned."""
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    async def fake_detect(self, msg):
        return TurnEndQuestionDetection(
            has_question=True,
            questions=(
                TurnEndQuestionItem(
                    question="Pick A or B?",
                    options=(
                        TurnEndQuestionOption(label="A"),
                        TurnEndQuestionOption(label="B"),
                    ),
                ),
            ),
        )

    monkeypatch.setattr(KimiSoul, "_detect_turn_end_question", fake_detect)

    sent_messages: list[object] = []

    def capturing_wire_send(msg):
        sent_messages.append(msg)
        # Auto-resolve question requests with an answer
        if isinstance(msg, QuestionRequest):
            msg.resolve({"Pick A or B?": "A"})

    monkeypatch.setattr(kimisoul_module, "wire_send", capturing_wire_send)

    # Set up wire context so get_wire_or_none() returns a wire
    wire = Wire()
    token = soul_module._current_wire.set(wire)
    try:
        outcome = TurnOutcome(
            stop_reason="no_tool_calls",
            final_message=Message(role="assistant", content="Should I do A or B?"),
            step_count=1,
        )
        result = await soul._maybe_ask_turn_end_question(outcome)
    finally:
        soul_module._current_wire.reset(token)

    assert result == "A"
    question_requests = [m for m in sent_messages if isinstance(m, QuestionRequest)]
    assert len(question_requests) == 1
    assert question_requests[0].questions[0].question == "Pick A or B?"
    assert question_requests[0].questions[0].body == "Should I do A or B?"


@pytest.mark.asyncio
async def test_maybe_ask_turn_end_question_dismissed(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the user dismisses the question, returns None."""
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    async def fake_detect(self, msg):
        return TurnEndQuestionDetection(
            has_question=True,
            questions=(
                TurnEndQuestionItem(
                    question="Pick?",
                    options=(
                        TurnEndQuestionOption(label="A"),
                        TurnEndQuestionOption(label="B"),
                    ),
                ),
            ),
        )

    monkeypatch.setattr(KimiSoul, "_detect_turn_end_question", fake_detect)

    def capturing_wire_send(msg):
        if isinstance(msg, QuestionRequest):
            # Resolve with empty answers = dismissed
            msg.resolve({})

    monkeypatch.setattr(kimisoul_module, "wire_send", capturing_wire_send)

    wire = Wire()
    token = soul_module._current_wire.set(wire)
    try:
        outcome = TurnOutcome(
            stop_reason="no_tool_calls",
            final_message=Message(role="assistant", content="Pick?"),
            step_count=1,
        )
        result = await soul._maybe_ask_turn_end_question(outcome)
    finally:
        soul_module._current_wire.reset(token)

    assert result is None


# -- Config toggle test --


@pytest.mark.asyncio
async def test_run_skips_detection_when_disabled(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When turn_end_question_detection is disabled, _maybe_ask_turn_end_question is not called."""
    runtime.config.loop_control.turn_end_question_detection = False

    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    detection_called = False

    async def fake_maybe_ask(self, outcome):
        nonlocal detection_called
        detection_called = True
        return None

    async def fake_turn(self, msg):
        return TurnOutcome(
            stop_reason="no_tool_calls",
            final_message=Message(role="assistant", content="Done."),
            step_count=1,
        )

    monkeypatch.setattr(KimiSoul, "_turn", fake_turn)
    monkeypatch.setattr(KimiSoul, "_maybe_ask_turn_end_question", fake_maybe_ask)
    monkeypatch.setattr(kimisoul_module, "wire_send", lambda msg: None)

    await soul.run("hello")

    assert not detection_called


@pytest.mark.asyncio
async def test_run_sends_follow_up_input_on_user_choice(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the user selects an option from a turn-end question, a FollowUpInput
    message is sent via wire so the TUI can display the user's choice."""
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    turn_call_count = 0

    async def fake_turn(self, msg):
        nonlocal turn_call_count
        turn_call_count += 1
        return TurnOutcome(
            stop_reason="no_tool_calls",
            final_message=Message(role="assistant", content="Pick A or B?"),
            step_count=1,
        )

    async def fake_maybe_ask(self, outcome):
        return "A"

    sent_messages: list[object] = []

    monkeypatch.setattr(KimiSoul, "_turn", fake_turn)
    monkeypatch.setattr(KimiSoul, "_maybe_ask_turn_end_question", fake_maybe_ask)
    monkeypatch.setattr(kimisoul_module, "wire_send", lambda msg: sent_messages.append(msg))

    await soul.run("hello")

    # A FollowUpInput should have been sent with the user's answer
    follow_ups = [m for m in sent_messages if isinstance(m, FollowUpInput)]
    assert len(follow_ups) == 1
    assert follow_ups[0].text == "A"
    # Two turns should have been run: the original + the follow-up
    assert turn_call_count == 2


# -- Retry on unparseable output test --


@pytest.mark.asyncio
async def test_detect_turn_end_question_retries_on_bad_json(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the first LLM response is not valid JSON, the detector retries
    and succeeds on the second attempt."""
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    call_count = 0

    async def fake_generate(*, chat_provider, system_prompt, tools, history):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # First attempt: garbage output
            return SimpleNamespace(message=Message(role="assistant", content="Sure, here is..."))
        # Second attempt: valid JSON
        return SimpleNamespace(
            message=Message(
                role="assistant",
                content=(
                    '{"has_question": true, "questions": [{"question": "A or B?",'
                    ' "options": [{"label": "A"}, {"label": "B"}]}]}'
                ),
            )
        )

    monkeypatch.setattr(kimisoul_module.kosong, "generate", fake_generate)

    assistant_msg = Message(role="assistant", content="Pick A or B?")
    result = await soul._detect_turn_end_question(assistant_msg)

    assert call_count == 2
    assert result is not None
    assert result.has_question is True


@pytest.mark.asyncio
async def test_detect_turn_end_question_gives_up_after_max_attempts(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When all retry attempts return unparseable output, detection returns None."""
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    call_count = 0

    async def fake_generate(*, chat_provider, system_prompt, tools, history):
        nonlocal call_count
        call_count += 1
        return SimpleNamespace(message=Message(role="assistant", content="I don't know"))

    monkeypatch.setattr(kimisoul_module.kosong, "generate", fake_generate)

    assistant_msg = Message(role="assistant", content="Pick A or B?")
    result = await soul._detect_turn_end_question(assistant_msg)

    assert call_count == soul._TURN_END_DETECT_MAX_ATTEMPTS
    assert result is None


@pytest.mark.asyncio
async def test_detect_turn_end_question_uses_heuristic_for_if_you_want_continue(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    async def fake_generate(*, chat_provider, system_prompt, tools, history):
        raise RuntimeError("detector unavailable")

    monkeypatch.setattr(kimisoul_module.kosong, "generate", fake_generate)

    assistant_msg = Message(
        role="assistant",
        content="前面的分析已经完成。\n\n如果你要，我可以继续直接做下去。",
    )
    result = await soul._detect_turn_end_question(assistant_msg)

    assert result is not None
    assert result.has_question is True
    assert result.questions[0].question == "要我继续吗？"
    assert result.questions[0].options[0].label == "继续"
    assert result.questions[0].options[1].label == "先别"


@pytest.mark.asyncio
async def test_detect_turn_end_question_does_not_override_detector_false_with_heuristic(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    async def fake_generate(*, chat_provider, system_prompt, tools, history):
        return SimpleNamespace(
            message=Message(role="assistant", content='{"has_question": false, "questions": []}')
        )

    monkeypatch.setattr(kimisoul_module.kosong, "generate", fake_generate)

    assistant_msg = Message(role="assistant", content="如果你要，我可以继续直接做下去。")
    result = await soul._detect_turn_end_question(assistant_msg)

    assert result is not None
    assert result.has_question is False


@pytest.mark.asyncio
async def test_detect_turn_end_question_uses_heuristic_for_if_continue(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    async def fake_generate(*, chat_provider, system_prompt, tools, history):
        raise RuntimeError("detector unavailable")

    monkeypatch.setattr(kimisoul_module.kosong, "generate", fake_generate)

    assistant_msg = Message(role="assistant", content="如果继续，我可以先处理 A。")
    result = await soul._detect_turn_end_question(assistant_msg)

    assert result is not None
    assert result.has_question is True
    assert result.questions[0].question == "要我继续吗？"
    assert result.questions[0].options[0].label == "继续"
    assert result.questions[0].options[1].label == "先别"


def test_heuristic_turn_end_question_ignores_quoted_example(
    runtime: Runtime,
    tmp_path: Path,
) -> None:
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    result = soul._heuristic_turn_end_question(
        "文案可以改成“如果你要，我可以继续直接做下去。”这种更自然的说法。"
    )

    assert result is None


@pytest.mark.asyncio
async def test_detect_turn_end_question_ignores_conditional_analysis_statement(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    async def fake_generate(*, chat_provider, system_prompt, tools, history):
        return SimpleNamespace(
            message=Message(role="assistant", content='{"has_question": false, "questions": []}')
        )

    monkeypatch.setattr(kimisoul_module.kosong, "generate", fake_generate)

    assistant_msg = Message(role="assistant", content="如果继续这样做，风险会更高。")
    result = await soul._detect_turn_end_question(assistant_msg)

    assert result is not None
    assert result.has_question is False


# -- Yes/No question detection tests --


class TestParseYesNoQuestion:
    def _make_soul(self, runtime: Runtime, tmp_path: Path) -> KimiSoul:
        return KimiSoul(
            Agent(
                name="Test",
                system_prompt="Test",
                toolset=EmptyToolset(),
                runtime=runtime,
            ),
            context=Context(file_backend=tmp_path / "history.jsonl"),
        )

    def test_parse_yes_no_question(self, runtime: Runtime, tmp_path: Path) -> None:
        """Yes/No questions should be recognized as valid choice questions."""
        soul = self._make_soul(runtime, tmp_path)
        result = soul._parse_turn_end_question_payload(
            '{"has_question": true, "questions": [{"question": "Should I proceed?",'
            ' "options": [{"label": "Yes", "description": "Continue with the change"},'
            ' {"label": "No", "description": "Cancel"}]}]}'
        )
        assert result is not None
        assert result.has_question is True
        assert len(result.questions) == 1
        assert result.questions[0].question == "Should I proceed?"
        assert result.questions[0].options[0].label == "Yes"
        assert result.questions[0].options[1].label == "No"


# -- Timeout tests --


@pytest.mark.asyncio
async def test_detect_turn_end_question_times_out(
    runtime: Runtime,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the side-channel LLM call exceeds the timeout, returns None."""
    soul = KimiSoul(
        Agent(
            name="Test",
            system_prompt="Test",
            toolset=EmptyToolset(),
            runtime=runtime,
        ),
        context=Context(file_backend=tmp_path / "history.jsonl"),
    )

    # Use a very short timeout for testing
    monkeypatch.setattr(KimiSoul, "_TURN_END_DETECT_TIMEOUT", 0.1)

    async def slow_generate(*, chat_provider, system_prompt, tools, history):
        await asyncio.sleep(10)  # much longer than the timeout
        return SimpleNamespace(
            message=Message(role="assistant", content='{"has_question": false, "questions": []}')
        )

    monkeypatch.setattr(kimisoul_module.kosong, "generate", slow_generate)

    assistant_msg = Message(role="assistant", content="Should I do A or B?")
    result = await soul._detect_turn_end_question(assistant_msg)

    assert result is None


def test_turn_end_question_prompt_mentions_continue_in_chinese() -> None:
    assert '"是否继续？"' in kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT


def test_turn_end_question_prompt_mentions_soft_permission_phrases() -> None:
    assert '"如果你愿意，我可以继续直接做下一轮。"' in (
        kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT
    )
    assert (
        'Chinese "是否 + action clause" / "如果你愿意，我可以..." / "如果你想，我可以..." / '
        '"如果你要，我可以..." / "如果继续，我可以..."'
        in kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT
    )
    assert "synthesize two concise options" in kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT


def test_turn_end_question_prompt_mentions_multiple_suggestions() -> None:
    assert '"如果继续这样做，风险会更高。"' in kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT
    assert (
        "pick from multiple concrete suggestions"
        in kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT
    )
    assert "Treat multiple concrete suggestions or recommended next steps as options" in (
        kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT
    )
    assert "without a literal question mark" in kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT
    assert "Mere recommendation lists or next-step suggestions" in (
        kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT
    )
    assert "Do not infer has_question=true from a numbered list alone" in (
        kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT
    )
    assert '"下一步我建议做 A、B、C，你想先做哪个？"' in (
        kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT
    )
    assert (
        '"Next steps: 1. Fix interactions 2. Improve performance 3. Tidy styling. '
        'Choose one for me to do first."' in kimisoul_module.TURN_END_QUESTION_DETECTOR_PROMPT
    )

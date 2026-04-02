from __future__ import annotations

from inline_snapshot import snapshot
from kosong.chat_provider import TokenUsage
from kosong.message import Message

import kimi_cli.prompts as prompts
from kimi_cli.soul.compaction import CompactionResult, SimpleCompaction, should_auto_compact
from kimi_cli.wire.types import TextPart, ThinkPart


def test_prepare_returns_original_when_not_enough_messages():
    messages = [Message(role="user", content=[TextPart(text="Only one message")])]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result == snapshot(
        SimpleCompaction.PrepareResult(
            compact_message=None,
            to_preserve=[Message(role="user", content=[TextPart(text="Only one message")])],
        )
    )


def test_prepare_skips_compaction_with_only_preserved_messages():
    messages = [
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest reply")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result == snapshot(
        SimpleCompaction.PrepareResult(
            compact_message=None,
            to_preserve=[
                Message(role="user", content=[TextPart(text="Latest question")]),
                Message(role="assistant", content=[TextPart(text="Latest reply")]),
            ],
        )
    )


def test_prepare_builds_compact_message_and_preserves_tail():
    messages = [
        Message(role="system", content=[TextPart(text="System note")]),
        Message(
            role="user",
            content=[TextPart(text="Old question"), ThinkPart(think="Hidden thoughts")],
        ),
        Message(role="assistant", content=[TextPart(text="Old answer")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result.compact_message == snapshot(
        Message(
            role="user",
            content=[
                TextPart(text="## Message 1\nRole: system\nContent:\n"),
                TextPart(text="System note"),
                TextPart(text="\n" + prompts.COMPACT),
            ],
        )
    )
    assert result.to_preserve == snapshot(
        [
            Message(
                role="user",
                content=[TextPart(text="Old question"), ThinkPart(think="Hidden thoughts")],
            ),
            Message(role="assistant", content=[TextPart(text="Old answer")]),
            Message(role="user", content=[TextPart(text="Latest question")]),
            Message(role="assistant", content=[TextPart(text="Latest answer")]),
        ]
    )


# --- CompactionResult.estimated_token_count tests ---


def test_estimated_token_count_with_usage_uses_output_tokens_for_summary():
    """When usage is available, the summary (first message) uses exact output tokens
    and preserved messages (remaining) use character-based estimation."""
    summary_msg = Message(role="user", content=[TextPart(text="compacted summary")])
    preserved_msg = Message(
        role="user",
        content=[TextPart(text="a" * 80)],  # 80 chars → 20 tokens
    )
    usage = TokenUsage(input_other=1000, output=150, input_cache_read=0)

    result = CompactionResult(messages=[summary_msg, preserved_msg], usage=usage)

    # 150 (exact output tokens) + 80//4 (text estimate) + 4 (per-message overhead)
    assert result.estimated_token_count == 150 + 20 + 4


def test_estimated_token_count_without_usage_estimates_all_from_text():
    """Without usage (no LLM call), all messages are estimated from text content."""
    messages = [
        Message(role="user", content=[TextPart(text="a" * 100)]),
        Message(role="assistant", content=[TextPart(text="b" * 200)]),
    ]
    result = CompactionResult(messages=messages, usage=None)

    # (100//4 + 4) + (200//4 + 4) = 29 + 54 = 83
    assert result.estimated_token_count == 300 // 4 + 2 * 4


def test_estimated_token_count_ignores_non_text_parts():
    """Non-text parts (think, etc.) should not inflate the estimate."""
    messages = [
        Message(
            role="user",
            content=[
                TextPart(text="a" * 40),
                ThinkPart(think="internal reasoning " * 100),
            ],
        ),
    ]
    result = CompactionResult(messages=messages, usage=None)

    # 40//4 + 4 (per-message overhead); ThinkPart is not counted
    assert result.estimated_token_count == 40 // 4 + 4


def test_estimated_token_count_empty_messages():
    """Empty message list should return 0."""
    result = CompactionResult(messages=[], usage=None)
    assert result.estimated_token_count == 0


def test_prepare_appends_custom_instruction():
    messages = [
        Message(role="user", content=[TextPart(text="Earliest question")]),
        Message(role="assistant", content=[TextPart(text="Earliest answer")]),
        Message(role="user", content=[TextPart(text="Old question")]),
        Message(role="assistant", content=[TextPart(text="Old answer")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(
        messages, custom_instruction="Preserve all discussions about the database"
    )

    assert result.compact_message is not None
    parts = result.compact_message.content
    last_part = parts[-1]
    assert isinstance(last_part, TextPart)
    # Custom instruction should be merged into the same TextPart as the COMPACT prompt
    assert last_part.text.startswith("\n" + prompts.COMPACT)
    assert "User's Custom Compaction Instruction" in last_part.text
    assert "Preserve all discussions about the database" in last_part.text


def test_prepare_without_custom_instruction_unchanged():
    """When no custom_instruction is given, the compact message should end with the COMPACT prompt."""
    messages = [
        Message(role="user", content=[TextPart(text="Earliest question")]),
        Message(role="assistant", content=[TextPart(text="Earliest answer")]),
        Message(role="user", content=[TextPart(text="Old question")]),
        Message(role="assistant", content=[TextPart(text="Old answer")]),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result.compact_message is not None
    parts = result.compact_message.content
    last_part = parts[-1]
    assert isinstance(last_part, TextPart)
    assert last_part.text == "\n" + prompts.COMPACT


def test_prepare_empty_messages():
    """prepare([]) returns (None, [])."""
    result = SimpleCompaction(max_preserved_messages=2).prepare([])

    assert result.compact_message is None
    assert list(result.to_preserve) == []


def test_prepare_max_preserved_zero():
    """prepare with max_preserved_messages=0 returns (None, messages)."""
    messages = [
        Message(role="user", content=[TextPart(text="Hello")]),
        Message(role="assistant", content=[TextPart(text="Hi")]),
    ]

    result = SimpleCompaction(max_preserved_messages=0).prepare(messages)

    assert result.compact_message is None
    assert list(result.to_preserve) == messages


def test_prepare_preserves_complete_turn_with_tool_messages():
    """When the preserved region includes an assistant(tool_calls) followed by
    tool messages, the complete turn (assistant+tool) is kept together in the
    preserved tail. Verify tool messages are included in preserved."""
    from kosong.message import ToolCall

    tc = ToolCall(
        id="call_abc",
        type="function",
        function=ToolCall.FunctionBody(name="grep", arguments='{"q":"x"}'),
    )
    messages = [
        Message(role="user", content=[TextPart(text="Earliest")]),
        Message(role="assistant", content=[TextPart(text="Earliest reply")]),
        Message(role="user", content=[TextPart(text="Middle question")]),
        Message(role="assistant", content=[TextPart(text="Middle reply")]),
        Message(
            role="assistant",
            content=[TextPart(text="Let me search")],
            tool_calls=[tc],
        ),
        Message(
            role="tool",
            content=[TextPart(text="grep result")],
            tool_call_id="call_abc",
        ),
        Message(role="user", content=[TextPart(text="Latest question")]),
        Message(role="assistant", content=[TextPart(text="Latest answer")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result.compact_message is not None
    preserved_roles = [(m.role, m.tool_call_id) for m in result.to_preserve]
    # The assistant(tool_calls) and its tool result are both preserved
    # as part of the complete turn alongside the 2 user turns.
    assert preserved_roles == [
        ("user", None),  # Middle question
        ("assistant", None),  # Middle reply
        ("assistant", None),  # assistant with tool_calls
        ("tool", "call_abc"),  # tool result
        ("user", None),  # Latest question
        ("assistant", None),  # Latest answer
    ]
    # Verify the tool message content is in preserved, not compacted
    tool_msgs = [m for m in result.to_preserve if m.role == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0].tool_call_id == "call_abc"


def test_prepare_includes_tool_calls_in_compact_message():
    """Verify that when an assistant message has tool_calls, the compact_message
    includes the serialized tool calls."""
    from kosong.message import ToolCall

    tc = ToolCall(
        id="call_xyz",
        type="function",
        function=ToolCall.FunctionBody(name="read_file", arguments='{"path":"a.py"}'),
    )
    messages = [
        Message(role="user", content=[TextPart(text="Old question")]),
        Message(
            role="assistant",
            content=[TextPart(text="Reading file")],
            tool_calls=[tc],
        ),
        Message(
            role="tool",
            content=[TextPart(text="file contents")],
            tool_call_id="call_xyz",
        ),
        Message(role="user", content=[TextPart(text="Next question")]),
        Message(role="assistant", content=[TextPart(text="Next answer")]),
        Message(role="user", content=[TextPart(text="Latest")]),
        Message(role="assistant", content=[TextPart(text="Latest reply")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result.compact_message is not None
    text_parts = [p.text for p in result.compact_message.content if isinstance(p, TextPart)]
    full_text = "".join(text_parts)
    assert "Tool Call: read_file(" in full_text


def test_prepare_includes_tool_call_id_for_tool_messages():
    """Verify tool role messages with tool_call_id show
    'tool (call_id: xxx)' in compact_message."""
    from kosong.message import ToolCall

    tc = ToolCall(
        id="call_123",
        type="function",
        function=ToolCall.FunctionBody(name="bash", arguments='{"cmd":"ls"}'),
    )
    messages = [
        Message(role="user", content=[TextPart(text="Old question")]),
        Message(
            role="assistant",
            content=[TextPart(text="Running command")],
            tool_calls=[tc],
        ),
        Message(
            role="tool",
            content=[TextPart(text="file_a\nfile_b")],
            tool_call_id="call_123",
        ),
        Message(role="user", content=[TextPart(text="Next")]),
        Message(role="assistant", content=[TextPart(text="Next reply")]),
        Message(role="user", content=[TextPart(text="Latest")]),
        Message(role="assistant", content=[TextPart(text="Latest reply")]),
    ]

    result = SimpleCompaction(max_preserved_messages=2).prepare(messages)

    assert result.compact_message is not None
    text_parts = [p.text for p in result.compact_message.content if isinstance(p, TextPart)]
    full_text = "".join(text_parts)
    assert "tool (call_id: call_123)" in full_text


# --- should_auto_compact tests ---


class TestShouldAutoCompact:
    """Test the auto-compaction trigger logic across different model context sizes."""

    def test_200k_model_triggers_by_reserved(self):
        """200K model with default config: reserved (50K) fires first at 150K (75%)."""
        # At 150K tokens: ratio check = 150K >= 170K (False), reserved check = 200K >= 200K (True)
        assert should_auto_compact(
            150_000, 200_000, trigger_ratio=0.85, reserved_context_size=50_000
        )

    def test_200k_model_below_threshold(self):
        """200K model: 140K tokens should NOT trigger (below both thresholds)."""
        assert not should_auto_compact(
            140_000, 200_000, trigger_ratio=0.85, reserved_context_size=50_000
        )

    def test_1m_model_triggers_by_ratio(self):
        """1M model with default config: ratio (85%) fires first at 850K."""
        # At 850K tokens: ratio check = 850K >= 850K (True)
        assert should_auto_compact(
            850_000, 1_000_000, trigger_ratio=0.85, reserved_context_size=50_000
        )

    def test_1m_model_below_ratio_threshold(self):
        """1M model: 840K tokens should NOT trigger (below 85% ratio, well above reserved)."""
        assert not should_auto_compact(
            840_000, 1_000_000, trigger_ratio=0.85, reserved_context_size=50_000
        )

    def test_custom_ratio_triggers_earlier(self):
        """Custom ratio=0.7 triggers at 70% of context."""
        # 200K * 0.7 = 140K
        assert should_auto_compact(
            140_000, 200_000, trigger_ratio=0.7, reserved_context_size=50_000
        )
        assert not should_auto_compact(
            139_999, 200_000, trigger_ratio=0.7, reserved_context_size=50_000
        )

    def test_zero_tokens_never_triggers(self):
        """Empty context should never trigger compaction."""
        assert not should_auto_compact(0, 200_000, trigger_ratio=0.85, reserved_context_size=50_000)

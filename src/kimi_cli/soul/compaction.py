from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol, runtime_checkable

import kosong
from kosong.chat_provider import ChatProviderError, TokenUsage
from kosong.message import Message
from kosong.tooling.empty import EmptyToolset

import kimi_cli.prompts as prompts
from kimi_cli.llm import LLM
from kimi_cli.soul.compaction_archive import stringify_tool_calls
from kimi_cli.soul.message import internal_user_message, system
from kimi_cli.utils.logging import logger
from kimi_cli.wire.types import (
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    TextPart,
    ThinkPart,
    VideoURLPart,
)


class CompactionResult(NamedTuple):
    messages: Sequence[Message]
    usage: TokenUsage | None

    @property
    def estimated_token_count(self) -> int:
        """Estimate the token count of the compacted messages.

        When LLM usage is available, ``usage.output`` gives the exact token count
        of the generated summary (the first message).  Preserved messages (all
        subsequent messages) are estimated from their text length.

        When usage is not available (no compaction LLM call was made), all
        messages are estimated from text length.

        The estimate is intentionally conservative — it will be replaced by the
        real value on the next LLM call.
        """
        if self.usage is not None and len(self.messages) > 0:
            summary_tokens = self.usage.output
            preserved_tokens = estimate_text_tokens(self.messages[1:])
            return summary_tokens + preserved_tokens

        return estimate_text_tokens(self.messages)


def estimate_text_tokens(messages: Sequence[Message]) -> int:
    """Estimate tokens from message content using heuristics.

    - Text: ~4 chars per token (conservative for English; underestimates CJK)
    - Per-message overhead: +4 tokens (role, structural tokens)
    - Image parts: +85 tokens each (typical for low-detail)
    - Audio/Video parts: +200 tokens each (rough estimate)
    """
    total = 0
    for msg in messages:
        total += 4  # per-message overhead
        for part in msg.content:
            if isinstance(part, TextPart):
                total += len(part.text) // 4
            elif isinstance(part, ImageURLPart):
                total += 85
            elif isinstance(part, (AudioURLPart, VideoURLPart)):
                total += 200
    return total


def should_auto_compact(
    token_count: int,
    max_context_size: int,
    *,
    trigger_ratio: float,
    reserved_context_size: int,
) -> bool:
    """Determine whether auto-compaction should be triggered.

    Returns True when either condition is met (whichever fires first):
    - Ratio-based: token_count >= max_context_size * trigger_ratio
    - Reserved-based: token_count + reserved_context_size >= max_context_size
    """
    return (
        token_count >= max_context_size * trigger_ratio
        or token_count + reserved_context_size >= max_context_size
    )


@runtime_checkable
class Compaction(Protocol):
    async def compact(
        self, messages: Sequence[Message], llm: LLM, *, custom_instruction: str = ""
    ) -> CompactionResult:
        """
        Compact a sequence of messages into a new sequence of messages.

        Args:
            messages (Sequence[Message]): The messages to compact.
            llm (LLM): The LLM to use for compaction.
            custom_instruction: Optional user instruction to guide compaction focus.

        Returns:
            CompactionResult: The compacted messages and token usage from the compaction LLM call.

        Raises:
            ChatProviderError: When the chat provider returns an error.
        """
        ...


if TYPE_CHECKING:

    def type_check(simple: SimpleCompaction):
        _: Compaction = simple


class SimpleCompaction:
    def __init__(self, max_preserved_messages: int = 2) -> None:
        self.max_preserved_messages = max_preserved_messages

    async def compact(
        self, messages: Sequence[Message], llm: LLM, *, custom_instruction: str = ""
    ) -> CompactionResult:
        compact_message, to_preserve = self.prepare(messages, custom_instruction=custom_instruction)
        if compact_message is None:
            return CompactionResult(messages=to_preserve, usage=None)

        # Call kosong.step to get the compacted context
        # Cap summary length to prevent overly long/short summaries
        max_output_tokens = max(4000, llm.max_context_size // 5)
        logger.debug(
            "Compacting context (max_output_tokens={max_output})",
            max_output=max_output_tokens,
        )
        # NOTE: kosong.step does not yet support max_tokens; log for now.
        result = await kosong.step(
            chat_provider=llm.chat_provider,
            system_prompt=(
                "You are a specialist in compacting agent conversation "
                "context. Produce structured, information-dense summaries "
                "that preserve all actionable details, file paths, "
                "decisions, and ongoing task state. Follow the exact "
                "output format specified in the user instructions."
            ),
            toolset=EmptyToolset(),
            history=[compact_message],
        )
        if result.usage:
            logger.debug(
                "Compaction used {input} input tokens and {output} output tokens",
                input=result.usage.input,
                output=result.usage.output,
            )

        content: list[ContentPart] = [
            system("Previous context has been compacted. Here is the compaction output:")
        ]
        compacted_msg = result.message

        # drop thinking parts if any
        content.extend(part for part in compacted_msg.content if not isinstance(part, ThinkPart))

        # Guard: ensure the LLM produced meaningful content (skip our system prefix)
        if not any(p.text.strip() for p in compacted_msg.content if isinstance(p, TextPart)):
            raise ChatProviderError("Compaction produced empty summary")

        compacted_messages: list[Message] = [internal_user_message(content)]
        compacted_messages.extend(to_preserve)
        return CompactionResult(messages=compacted_messages, usage=result.usage)

    class PrepareResult(NamedTuple):
        compact_message: Message | None
        to_preserve: Sequence[Message]

    def prepare(
        self, messages: Sequence[Message], *, custom_instruction: str = ""
    ) -> PrepareResult:
        if not messages or self.max_preserved_messages <= 0:
            return self.PrepareResult(compact_message=None, to_preserve=messages)

        preserve_start_index = len(messages)

        # Walk backward counting user messages to find preserve boundary.
        # A "turn" is: user → assistant → trailing tool messages.
        n_user = 0
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role == "user":
                n_user += 1
                if n_user == self.max_preserved_messages:
                    preserve_start_index = index
                    break

        if n_user < self.max_preserved_messages:
            return self.PrepareResult(compact_message=None, to_preserve=messages)

        # Walk preserve_start_index backward to include any preceding
        # assistant message whose tool_calls have results in the preserved
        # region, so we don't orphan tool calls from their results.
        while preserve_start_index > 0:
            prev = messages[preserve_start_index - 1]
            if prev.role == "assistant" and prev.tool_calls:
                preserve_start_index -= 1
            else:
                break

        to_compact = messages[:preserve_start_index]
        to_preserve = messages[preserve_start_index:]

        if not to_compact:
            # Let's hope this won't exceed the context size limit
            return self.PrepareResult(compact_message=None, to_preserve=to_preserve)

        # Create input message for compaction
        compact_message = Message(role="user", content=[])
        for i, msg in enumerate(to_compact):
            role_label = msg.role
            if msg.role == "tool" and msg.tool_call_id:
                role_label = f"tool (call_id: {msg.tool_call_id})"
            compact_message.content.append(
                TextPart(text=f"## Message {i + 1}\nRole: {role_label}\nContent:\n")
            )
            compact_message.content.extend(
                part for part in msg.content if not isinstance(part, ThinkPart)
            )
            if msg.tool_calls:
                compact_message.content.append(
                    TextPart(text=f"Tool calls: {stringify_tool_calls(msg.tool_calls)}")
                )
        prompt_text = "\n" + prompts.COMPACT
        if custom_instruction:
            prompt_text += (
                "\n\n**User's Custom Compaction Instruction:**\n"
                "The user has specifically requested the following focus during compaction. "
                "You MUST prioritize this instruction above the default compression priorities:\n"
                f"{custom_instruction}"
            )
        compact_message.content.append(TextPart(text=prompt_text))
        return self.PrepareResult(compact_message=compact_message, to_preserve=to_preserve)

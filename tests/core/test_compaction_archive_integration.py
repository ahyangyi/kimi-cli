from __future__ import annotations

from pathlib import Path

from kosong.chat_provider import TokenUsage
from kosong.message import Message

from kimi_cli.soul.agent import Agent, Runtime
from kimi_cli.soul.compaction import CompactionResult
from kimi_cli.soul.compaction_archive import load_compaction_archives
from kimi_cli.soul.context import Context
from kimi_cli.soul.kimisoul import KimiSoul
from kimi_cli.soul.message import internal_user_message, system
from kimi_cli.soul.toolset import KimiToolset
from kimi_cli.tools.context import RecallCompactedContext
from kimi_cli.tools.context.recall_compacted import Params
from kimi_cli.wire.types import TextPart


class FakeCompaction:
    async def compact(self, messages, llm, *, custom_instruction: str = "") -> CompactionResult:
        return CompactionResult(
            messages=[
                internal_user_message(
                    [
                        system(
                            "Previous context has been compacted. Here is the compaction output:"
                        ),
                        TextPart(
                            text=(
                                "<current_focus>\nInvestigating alpha issue\n</current_focus>\n"
                                "<important_context>\n- alpha traceback kept\n</important_context>"
                            )
                        ),
                    ]
                ),
                Message(role="user", content=[TextPart(text="Latest question")]),
                Message(role="assistant", content=[TextPart(text="Latest answer")]),
            ],
            usage=TokenUsage(input_other=120, output=40, input_cache_read=0),
        )


async def test_compaction_registers_archive_and_injects_recall_notice(
    runtime: Runtime, tmp_path: Path, monkeypatch
) -> None:
    recall_tool = RecallCompactedContext(runtime)
    toolset = KimiToolset()
    toolset.add(recall_tool)
    agent = Agent(
        name="Test Agent",
        system_prompt="Test system prompt.",
        toolset=toolset,
        runtime=runtime,
    )
    context = Context(file_backend=tmp_path / "history.jsonl")
    await context.append_message(
        [
            Message(role="user", content=[TextPart(text="Old alpha traceback")]),
            Message(role="assistant", content=[TextPart(text="Old alpha analysis")]),
            Message(role="user", content=[TextPart(text="Latest question")]),
            Message(role="assistant", content=[TextPart(text="Latest answer")]),
        ]
    )

    soul = KimiSoul(agent, context=context)
    soul._compaction = FakeCompaction()  # type: ignore[assignment]
    monkeypatch.setattr("kimi_cli.soul.kimisoul.wire_send", lambda _msg: None)

    assert [tool.name for tool in toolset.tools] == []

    await soul.compact_context()

    records = load_compaction_archives(context.file_backend)
    assert len(records) == 1
    assert records[0].id == "c001"
    assert "Investigating alpha issue" in records[0].summary
    assert any(
        any(
            isinstance(part, TextPart) and "RecallCompactedContext tool" in part.text
            for part in message.content
        )
        for message in context.history
    )
    assert [tool.name for tool in toolset.tools] == ["RecallCompactedContext"]
    assert context.token_count > 0

    recall_result = await recall_tool(Params(query="alpha"))
    assert not recall_result.is_error
    assert "Old alpha traceback" in recall_result.output


async def test_recall_tool_is_visible_immediately_when_archives_already_exist(
    runtime: Runtime, tmp_path: Path
) -> None:
    context = Context(file_backend=tmp_path / "history.jsonl")
    context.file_backend.touch()
    archive_file = tmp_path / "history_1.jsonl"
    archive_file.write_text(
        Message(role="user", content=[TextPart(text="Earlier compacted note")]).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    from kimi_cli.soul.compaction_archive import register_compaction_archive

    register_compaction_archive(
        context.file_backend,
        archive_file,
        message_count=1,
        summary="Earlier compacted note",
    )

    recall_tool = RecallCompactedContext(runtime)
    toolset = KimiToolset()
    toolset.add(recall_tool)
    soul = KimiSoul(
        Agent(
            name="Test Agent",
            system_prompt="Test system prompt.",
            toolset=toolset,
            runtime=runtime,
        ),
        context=context,
    )

    assert soul is not None
    assert [tool.name for tool in toolset.tools] == ["RecallCompactedContext"]

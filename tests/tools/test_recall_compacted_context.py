from __future__ import annotations

from pathlib import Path

from inline_snapshot import snapshot
from kosong.message import Message

from kimi_cli.soul.compaction_archive import register_compaction_archive
from kimi_cli.soul.message import internal_user_message, system
from kimi_cli.tools.context.recall_compacted import Params, RecallCompactedContext
from kimi_cli.wire.types import TextPart, ThinkPart


def _write_archive(archive_file: Path, messages: list[Message]) -> None:
    archive_file.parent.mkdir(parents=True, exist_ok=True)
    with archive_file.open("w", encoding="utf-8") as f:
        for message in messages:
            f.write(message.model_dump_json(exclude_none=True) + "\n")


async def test_list_archives_returns_summary(
    recall_compacted_context_tool: RecallCompactedContext, tmp_path: Path
) -> None:
    context_file = tmp_path / "context.jsonl"
    context_file.touch()
    archive_file = tmp_path / "context_1.jsonl"
    _write_archive(
        archive_file,
        [
            Message(role="user", content=[TextPart(text="Investigate foo.py")]),
            Message(role="assistant", content=[TextPart(text="Root cause was a bad path")]),
        ],
    )
    registration = register_compaction_archive(
        context_file,
        archive_file,
        message_count=2,
        summary="foo.py path investigation",
    )
    recall_compacted_context_tool.bind_context_file(lambda: context_file)

    result = await recall_compacted_context_tool(Params())

    assert not result.is_error
    assert result.output == snapshot(
        "Available compacted-context archives for this trajectory:\n"
        f"- c001 | {registration.record.created_at} | 2 messages | foo.py path investigation\n"
    )
    assert result.message == snapshot("Listed 1 compacted-context archive(s).")


async def test_search_returns_matching_excerpt_and_sanitizes_tags(
    recall_compacted_context_tool: RecallCompactedContext, tmp_path: Path
) -> None:
    context_file = tmp_path / "context.jsonl"
    context_file.touch()
    archive_file = tmp_path / "context_1.jsonl"
    _write_archive(
        archive_file,
        [
            internal_user_message(
                [system("Previous note about foo.py"), TextPart(text="Keep this")]
            ),
            Message(
                role="user",
                content=[
                    TextPart(text="Please inspect foo.py for the EACCES error"),
                    ThinkPart(think="private chain of thought"),
                ],
            ),
            Message(
                role="assistant", content=[TextPart(text="The shell command failed with EACCES")]
            ),
        ],
    )
    register_compaction_archive(
        context_file,
        archive_file,
        message_count=3,
        summary="foo.py EACCES troubleshooting",
    )
    recall_compacted_context_tool.bind_context_file(lambda: context_file)

    result = await recall_compacted_context_tool(Params(query="foo.py"))

    assert not result.is_error
    assert "<system>" not in result.output
    assert "private chain of thought" not in result.output
    assert result.output == snapshot(
        """\
Excerpt 1 | c001 | score 6 | messages 1-2
Summary: foo.py EACCES troubleshooting
[1] user [internal]
[system] Previous note about foo.py
Keep this

[2] user
Please inspect foo.py for the EACCES error

"""
    )
    assert result.message == snapshot("Found 1 excerpt(s) across 1 archive(s) for query `foo.py`.")


async def test_missing_archive_id_returns_error(
    recall_compacted_context_tool: RecallCompactedContext, tmp_path: Path
) -> None:
    context_file = tmp_path / "context.jsonl"
    context_file.touch()
    archive_file = tmp_path / "context_1.jsonl"
    _write_archive(archive_file, [Message(role="user", content=[TextPart(text="hello")])])
    register_compaction_archive(
        context_file,
        archive_file,
        message_count=1,
        summary="hello",
    )
    recall_compacted_context_tool.bind_context_file(lambda: context_file)

    result = await recall_compacted_context_tool(Params(archive_id="c999"))

    assert result.is_error
    assert result.message == snapshot(
        "Compacted-context archive `c999` was not found for the current conversation trajectory."
    )


async def test_no_archives_returns_informational_message(
    recall_compacted_context_tool: RecallCompactedContext, tmp_path: Path
) -> None:
    """When no archives exist, the tool returns an informational message (not an error)."""
    context_file = tmp_path / "context.jsonl"
    context_file.touch()
    recall_compacted_context_tool.bind_context_file(lambda: context_file)

    result = await recall_compacted_context_tool(Params())

    assert not result.is_error
    assert result.output == snapshot(
        "No compacted-context archives are available for this conversation trajectory.\n"
    )
    assert result.message == snapshot("No compacted context archives found.")


async def test_search_no_matches_returns_archive_list(
    recall_compacted_context_tool: RecallCompactedContext, tmp_path: Path
) -> None:
    """When query has no matches, returns 'No matching excerpts' plus the archive list."""
    context_file = tmp_path / "context.jsonl"
    context_file.touch()
    archive_file = tmp_path / "context_1.jsonl"
    _write_archive(
        archive_file,
        [
            Message(role="user", content=[TextPart(text="Talk about apples")]),
            Message(
                role="assistant",
                content=[TextPart(text="Apples are red fruit")],
            ),
        ],
    )
    register_compaction_archive(
        context_file,
        archive_file,
        message_count=2,
        summary="Discussion about apples",
    )
    recall_compacted_context_tool.bind_context_file(lambda: context_file)

    result = await recall_compacted_context_tool(Params(query="zzz_nonexistent_term"))

    assert not result.is_error
    assert "No matching excerpts" in result.output
    assert "c001" in result.output
    assert "Discussion about apples" in result.output
    assert result.message == snapshot("No compacted-context matches found.")


async def test_missing_archive_file_gracefully_handled(
    recall_compacted_context_tool: RecallCompactedContext, tmp_path: Path
) -> None:
    """When an archive file referenced in manifest doesn't exist on disk,
    search handles it gracefully (doesn't crash)."""
    context_file = tmp_path / "context.jsonl"
    context_file.touch()
    archive_file = tmp_path / "context_1.jsonl"
    # Register archive but don't write the file — simulate missing file
    register_compaction_archive(
        context_file,
        archive_file,
        message_count=1,
        summary="Ghost archive",
    )
    recall_compacted_context_tool.bind_context_file(lambda: context_file)

    # Should not crash — _search_single_archive checks archive_path.exists()
    result = await recall_compacted_context_tool(Params(query="anything"))

    assert not result.is_error
    assert "No matching excerpts" in result.output

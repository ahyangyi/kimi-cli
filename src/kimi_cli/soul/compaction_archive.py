from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from textwrap import shorten

from kosong.message import Message, ToolCall
from pydantic import BaseModel

from kimi_cli.utils.logging import logger
from kimi_cli.utils.turns import is_checkpoint_user_text, is_internal_user_message
from kimi_cli.wire.types import (
    AudioURLPart,
    ContentPart,
    ImageURLPart,
    TextPart,
    ThinkPart,
    VideoURLPart,
)

_MANIFEST_SUFFIX = ".compaction-archives.jsonl"
_SUMMARY_WIDTH = 280


class CompactionArchiveRecord(BaseModel):
    schema_version: int = 1
    id: str
    archive_file: str
    created_at: str
    message_count: int
    summary: str = ""


class ArchiveRegistrationResult(BaseModel):
    record: CompactionArchiveRecord
    total_archives: int


def manifest_path_for_context(context_file: Path) -> Path:
    return context_file.with_name(f"{context_file.stem}{_MANIFEST_SUFFIX}")


def load_compaction_archives(context_file: Path) -> list[CompactionArchiveRecord]:
    manifest_path = manifest_path_for_context(context_file)
    if not manifest_path.exists():
        return []

    records: list[CompactionArchiveRecord] = []
    with manifest_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(CompactionArchiveRecord.model_validate_json(line))
            except Exception:
                logger.warning(
                    "Skipping malformed compaction archive record in {path}:{line_no}",
                    path=manifest_path,
                    line_no=line_no,
                )
    return records


def register_compaction_archive(
    context_file: Path,
    archive_file: Path,
    *,
    message_count: int,
    summary: str,
) -> ArchiveRegistrationResult:
    records = load_compaction_archives(context_file)
    max_id = max((int(r.id[1:]) for r in records), default=0)
    record = CompactionArchiveRecord(
        id=f"c{max_id + 1:03d}",
        archive_file=archive_file.name,
        created_at=datetime.now().astimezone().isoformat(),
        message_count=message_count,
        summary=shorten(summary.strip(), width=_SUMMARY_WIDTH, placeholder="…") if summary else "",
    )

    # NOTE: Assumes single-writer — concurrent compactions on the same trajectory
    # are not expected.  If that changes, add file-level locking here.
    manifest_path = manifest_path_for_context(context_file)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(record.model_dump_json() + "\n")

    return ArchiveRegistrationResult(record=record, total_archives=len(records) + 1)


def resolve_compaction_archive_path(context_file: Path, record: CompactionArchiveRecord) -> Path:
    return context_file.parent / record.archive_file


def sanitize_archive_text(text: str) -> str:
    sanitized = text.replace("<system-reminder>", "[system-reminder]\n")
    sanitized = sanitized.replace("</system-reminder>", "")
    sanitized = sanitized.replace("<system-hint>", "[system-hint]\n")
    sanitized = sanitized.replace("</system-hint>", "")
    sanitized = sanitized.replace("<system>", "[system] ")
    sanitized = sanitized.replace("</system>", "")
    return sanitized.strip()


def stringify_content_for_archive(
    parts: Sequence[ContentPart], *, include_thinking: bool = False
) -> str:
    segments: list[str] = []
    for part in parts:
        match part:
            case TextPart(text=text):
                sanitized = sanitize_archive_text(text)
                if sanitized:
                    segments.append(sanitized)
            case ThinkPart(think=think):
                if include_thinking and think.strip():
                    segments.append(f"[thinking]\n{think.strip()}")
            case ImageURLPart():
                segments.append("[image]")
            case AudioURLPart():
                segments.append("[audio]")
            case VideoURLPart():
                segments.append("[video]")
            case _:
                segments.append(f"[{part.type}]")
    return "\n".join(segments)


def stringify_tool_calls(tool_calls: Sequence[ToolCall]) -> str:
    lines: list[str] = []
    for tc in tool_calls:
        args_raw = tc.function.arguments or "{}"
        try:
            args = json.loads(args_raw)
            args_text = json.dumps(args, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            args_text = args_raw
        lines.append(f"Tool Call: {tc.function.name}({args_text})")
    return "\n".join(lines)


def stringify_message_for_archive(message: Message, *, include_thinking: bool = False) -> str:
    segments: list[str] = []
    content_text = stringify_content_for_archive(message.content, include_thinking=include_thinking)
    if content_text:
        segments.append(content_text)
    if message.tool_calls:
        segments.append(stringify_tool_calls(message.tool_calls))
    return "\n".join(segments).strip()


def build_compaction_summary(messages: Sequence[Message]) -> str:
    if not messages:
        return ""
    prefix = "[system] Previous context has been compacted. Here is the compaction output:"
    for message in messages:
        text = stringify_message_for_archive(message)
        if not text:
            continue
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
        if text:
            return text
    return ""


def is_checkpoint_message(message: Message) -> bool:
    if message.role != "user":
        return False
    for part in message.content:
        if isinstance(part, TextPart):
            return is_checkpoint_user_text(part.text)
    return False


def load_archive_messages(archive_file: Path) -> list[Message]:
    messages: list[Message] = []
    with archive_file.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                logger.warning(
                    "Skipping malformed archive line in {path}:{line_no}",
                    path=archive_file,
                    line_no=line_no,
                )
                continue
            role = raw.get("role")
            if role in {"_usage", "_checkpoint"}:
                continue
            try:
                message = Message.model_validate(raw)
            except Exception:
                logger.warning(
                    "Skipping invalid archive message in {path}:{line_no}",
                    path=archive_file,
                    line_no=line_no,
                )
                continue
            if is_checkpoint_message(message):
                continue
            messages.append(message)
    return messages


def archive_role_label(message: Message) -> str:
    role = message.role
    if message.role == "user" and is_internal_user_message(message):
        role += " [internal]"
    if message.role == "tool" and message.tool_call_id:
        role += f" [{message.tool_call_id}]"
    return role

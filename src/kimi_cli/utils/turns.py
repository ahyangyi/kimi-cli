from __future__ import annotations

import re
from collections.abc import Mapping
from typing import cast

from kosong.message import Message

from kimi_cli.soul.message import INTERNAL_USER_NAME
from kimi_cli.wire.types import TextPart

CHECKPOINT_USER_PATTERN = re.compile(r"^<system>CHECKPOINT \d+</system>$")
_LEGACY_INTERNAL_USER_PREFIXES = (
    "<system-reminder>\nThe user sent a new reminder during the current turn.",
    "<system-reminder>\nPlan mode is active.",
    "<system-reminder>\nPlan mode still active (see full instructions earlier).",
    "<system>Reminder: the following skill suggestions may be helpful in the current context.",
    "<system>The user just ran `/init` slash command.",
    "<system>The user has added an additional directory to the workspace:",
    "<system>The user has imported context from ",
    "<system>Previous context has been compacted. Here is the compaction output:",
    "<system>You just got a D-Mail from your future self.",
)


def _first_text_part_message(message: Message) -> str | None:
    for part in message.content:
        if isinstance(part, TextPart):
            return part.text
    return None


def _first_text_part_record(record: Mapping[str, object]) -> str | None:
    content = record.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    for part in cast(list[object], content):
        if not isinstance(part, Mapping):
            continue
        text = cast(Mapping[str, object], part).get("text")
        if isinstance(text, str):
            return text
    return None


def is_checkpoint_user_text(text: str) -> bool:
    return CHECKPOINT_USER_PATTERN.fullmatch(text.strip()) is not None


def is_internal_user_text(text: str) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(prefix) for prefix in _LEGACY_INTERNAL_USER_PREFIXES)


def is_internal_user_message(message: Message) -> bool:
    if message.role != "user":
        return False
    if message.name == INTERNAL_USER_NAME:
        return True
    first_text = _first_text_part_message(message)
    return first_text is not None and is_internal_user_text(first_text)


def is_internal_user_record(record: Mapping[str, object]) -> bool:
    if record.get("role") != "user":
        return False
    if record.get("name") == INTERNAL_USER_NAME:
        return True
    first_text = _first_text_part_record(record)
    return first_text is not None and is_internal_user_text(first_text)


def is_real_user_turn_start_message(message: Message) -> bool:
    if message.role != "user":
        return False
    first_text = _first_text_part_message(message)
    if first_text is not None and is_checkpoint_user_text(first_text):
        return False
    return not is_internal_user_message(message)


def is_real_user_turn_start_record(record: Mapping[str, object]) -> bool:
    if record.get("role") != "user":
        return False
    first_text = _first_text_part_record(record)
    if first_text is not None and is_checkpoint_user_text(first_text):
        return False
    return not is_internal_user_record(record)

import asyncio
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import override

from kosong.message import Message
from kosong.tooling import CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field

from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.compaction_archive import (
    CompactionArchiveRecord,
    archive_role_label,
    load_archive_messages,
    load_compaction_archives,
    resolve_compaction_archive_path,
    stringify_message_for_archive,
)
from kimi_cli.tools.utils import ToolResultBuilder, load_desc
from kimi_cli.utils.logging import logger

MAX_RESULTS = 5
OUTPUT_MAX_CHARS = 12_000
EXCERPT_WINDOW = 1
_TOKEN_RE = re.compile(r"[\w./:-]+|[\u4e00-\u9fff]+")


class Params(BaseModel):
    query: str | None = Field(
        default=None,
        description=(
            "Targeted keywords to search for in compacted-context archives. Leave empty to list "
            "available archives and their summaries."
        ),
    )
    archive_id: str | None = Field(
        default=None,
        description=(
            "Optional archive ID like `c001` to restrict lookup to a single compacted-context "
            "archive."
        ),
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=MAX_RESULTS,
        description=f"Maximum number of excerpts to return. Defaults to {3}, max {MAX_RESULTS}.",
    )


@dataclass(frozen=True, slots=True)
class SearchHit:
    record: CompactionArchiveRecord
    score: int
    start_index: int
    end_index: int


class RecallCompactedContext(CallableTool2[Params]):
    name: str = "RecallCompactedContext"
    description: str = load_desc(Path(__file__).parent / "description.md")
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        super().__init__()
        self._context_file_getter: Callable[[], Path] = lambda: runtime.session.context_file

    def bind_context_file(self, getter: Callable[[], Path]) -> None:
        """Late-bind the current trajectory context file after KimiSoul is constructed."""
        self._context_file_getter = getter

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
        context_file = self._context_file_getter()
        records = load_compaction_archives(context_file)
        builder = ToolResultBuilder(max_chars=OUTPUT_MAX_CHARS)

        if not records:
            builder.write(
                "No compacted-context archives are available for this conversation trajectory.\n"
            )
            return builder.ok(message="No compacted context archives found", brief="No archives")

        selected_records = records
        if params.archive_id:
            selected_records = [record for record in records if record.id == params.archive_id]
            if not selected_records:
                return ToolError(
                    message=(
                        f"Compacted-context archive `{params.archive_id}` was not found for the "
                        "current conversation trajectory."
                    ),
                    brief="Archive not found",
                )

        if not params.query or not params.query.strip():
            self._render_archive_list(builder, selected_records)
            return builder.ok(
                message=f"Listed {len(selected_records)} compacted-context archive(s)",
                brief="Archive list",
            )

        query = params.query.strip()
        messages_cache: dict[str, list[Message]] = {}
        hits = await self._search_archives(selected_records, context_file, query, messages_cache)
        if not hits:
            builder.write(
                f"No matching excerpts were found for query `{query}` in the selected archives.\n\n"
            )
            self._render_archive_list(builder, selected_records)
            return builder.ok(message="No compacted-context matches found", brief="No matches")

        limited_hits = hits[: params.max_results]
        # Ensure archive messages are available (usually pre-cached during search)
        for hit in limited_hits:
            if hit.record.id not in messages_cache:
                archive_path = resolve_compaction_archive_path(context_file, hit.record)
                try:
                    messages_cache[hit.record.id] = load_archive_messages(archive_path)
                except FileNotFoundError:
                    logger.warning(
                        "Archive file missing for {archive_id}, skipping",
                        archive_id=hit.record.id,
                    )
        for index, hit in enumerate(limited_hits, start=1):
            if hit.record.id not in messages_cache:
                continue
            self._render_hit(builder, index, hit, messages_cache[hit.record.id])

        matched_archives = {hit.record.id for hit in limited_hits}
        return builder.ok(
            message=(
                f"Found {len(limited_hits)} excerpt(s) across {len(matched_archives)} archive(s) "
                f"for query `{query}`"
            ),
            brief="Archive excerpts",
        )

    async def _search_archives(
        self,
        records: Sequence[CompactionArchiveRecord],
        context_file: Path,
        query: str,
        messages_cache: dict[str, list[Message]],
    ) -> list[SearchHit]:
        tasks = [
            asyncio.to_thread(
                self._search_single_archive,
                record,
                context_file,
                query,
                messages_cache,
            )
            for record in records
        ]
        results = await asyncio.gather(*tasks)
        hits = [hit for archive_hits in results for hit in archive_hits]
        hits.sort(key=lambda hit: (-hit.score, hit.record.id, hit.start_index))
        return hits

    def _search_single_archive(
        self,
        record: CompactionArchiveRecord,
        context_file: Path,
        query: str,
        messages_cache: dict[str, list[Message]],
    ) -> list[SearchHit]:
        archive_path = resolve_compaction_archive_path(context_file, record)
        if not archive_path.exists():
            return []

        messages = load_archive_messages(archive_path)
        messages_cache[record.id] = messages
        if not messages:
            return []

        query_lc = query.lower()
        tokens = self._query_tokens(query_lc)
        raw_hits: list[tuple[int, int, int, int]] = []
        for index, message in enumerate(messages):
            text = stringify_message_for_archive(message)
            score = self._score_text(text, query_lc, tokens)
            if score <= 0:
                continue
            start = max(0, index - EXCERPT_WINDOW)
            end = min(len(messages), index + EXCERPT_WINDOW + 1)
            raw_hits.append((score, start, end, index))

        raw_hits.sort(key=lambda item: (-item[0], item[1], item[3]))
        accepted: list[SearchHit] = []
        covered_ranges: list[tuple[int, int]] = []
        for score, start, end, _match_index in raw_hits:
            if any(
                not (end <= existing_start or start >= existing_end)
                for existing_start, existing_end in covered_ranges
            ):
                continue
            accepted.append(SearchHit(record=record, score=score, start_index=start, end_index=end))
            covered_ranges.append((start, end))
        return accepted

    @staticmethod
    def _query_tokens(query: str) -> list[str]:
        raw = _TOKEN_RE.findall(query)
        # Filter single-char ASCII tokens; keep single CJK characters
        tokens = [t for t in raw if len(t) > 1 or "\u4e00" <= t <= "\u9fff"]
        if query and query not in tokens:
            tokens.append(query)
        return list(dict.fromkeys(tokens))

    @staticmethod
    def _score_text(text: str, query: str, tokens: Sequence[str]) -> int:
        lowered = text.lower()
        if not lowered.strip():
            return 0
        score = 0
        if query in lowered:
            score += max(3, len(tokens) + 1)
            # Bonus for word-boundary match (query appears as a whole word)
            if re.search(r"(?:^|\W)" + re.escape(query) + r"(?:\W|$)", lowered):
                score += 2
        score += sum(1 for token in tokens if token in lowered)
        return score

    @staticmethod
    def _render_archive_list(
        builder: ToolResultBuilder, records: Sequence[CompactionArchiveRecord]
    ) -> None:
        builder.write("Available compacted-context archives for this trajectory:\n")
        for record in records:
            line = f"- {record.id} | {record.created_at} | {record.message_count} messages"
            if record.summary:
                line += f" | {record.summary}"
            builder.write(line + "\n")

    @staticmethod
    def _render_hit(
        builder: ToolResultBuilder,
        index: int,
        hit: SearchHit,
        messages: Sequence[Message],
    ) -> None:
        builder.write(
            "Excerpt "
            f"{index} | {hit.record.id} | score {hit.score} | messages "
            f"{hit.start_index + 1}-{hit.end_index}\n"
        )
        if hit.record.summary:
            builder.write(f"Summary: {hit.record.summary}\n")
        for message_index in range(hit.start_index, hit.end_index):
            message = messages[message_index]
            label = archive_role_label(message)
            text = stringify_message_for_archive(message)
            builder.write(f"[{message_index + 1}] {label}\n")
            if text:
                builder.write(text + "\n")
            builder.write("\n")

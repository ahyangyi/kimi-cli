import difflib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Literal, override

from kaos.path import KaosPath
from kosong.tooling import CallableTool2, ToolError, ToolReturnValue
from pydantic import BaseModel, Field, ValidationInfo, field_validator

from kimi_cli.soul.agent import Runtime
from kimi_cli.soul.approval import Approval
from kimi_cli.tools.display import DisplayBlock
from kimi_cli.tools.file import FileActions
from kimi_cli.tools.file.plan_mode import inspect_plan_edit_target
from kimi_cli.tools.utils import load_desc
from kimi_cli.utils.diff import build_diff_blocks
from kimi_cli.utils.path import is_within_workspace

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class ReplaceOp(BaseModel):
    kind: Literal["replace"] = "replace"
    old: str = Field(description="The old string to replace. Can be multi-line.")
    new: str = Field(description="The new string to replace with. Can be multi-line.")
    replace_all: bool = Field(description="Whether to replace all occurrences.", default=False)

    @field_validator("old")
    @classmethod
    def validate_old(cls, value: str) -> str:
        if not value:
            raise ValueError("old cannot be empty")
        return value


class AppendOp(BaseModel):
    kind: Literal["append"] = "append"
    content: str = Field(description="The content to append to the end of the file.")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value:
            raise ValueError("content cannot be empty")
        return value


class PrependOp(BaseModel):
    kind: Literal["prepend"] = "prepend"
    content: str = Field(description="The content to insert at the beginning of the file.")

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value:
            raise ValueError("content cannot be empty")
        return value


class DeleteOp(BaseModel):
    kind: Literal["delete"] = "delete"
    old: str = Field(description="The string to delete from the file. Can be multi-line.")
    replace_all: bool = Field(description="Whether to delete all occurrences.", default=False)

    @field_validator("old")
    @classmethod
    def validate_old(cls, value: str) -> str:
        if not value:
            raise ValueError("old cannot be empty")
        return value


class InsertBeforeOp(BaseModel):
    kind: Literal["insert_before"] = "insert_before"
    anchor: str = Field(description="Insert the content before this anchor string.")
    content: str = Field(description="The content to insert.")
    occurrence: int = Field(
        description=(
            "Which anchor occurrence to target. Positive values count from the beginning; "
            "negative values count backward from the end."
        ),
        default=1,
    )

    @field_validator("anchor", "content")
    @classmethod
    def validate_strings(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            raise ValueError(f"{info.field_name} cannot be empty")
        return value

    @field_validator("occurrence")
    @classmethod
    def validate_occurrence(cls, value: int) -> int:
        if value == 0:
            raise ValueError("occurrence cannot be 0")
        return value


class InsertAfterOp(BaseModel):
    kind: Literal["insert_after"] = "insert_after"
    anchor: str = Field(description="Insert the content after this anchor string.")
    content: str = Field(description="The content to insert.")
    occurrence: int = Field(
        description=(
            "Which anchor occurrence to target. Positive values count from the beginning; "
            "negative values count backward from the end."
        ),
        default=1,
    )

    @field_validator("anchor", "content")
    @classmethod
    def validate_strings(cls, value: str, info: ValidationInfo) -> str:
        if not value:
            raise ValueError(f"{info.field_name} cannot be empty")
        return value

    @field_validator("occurrence")
    @classmethod
    def validate_occurrence(cls, value: int) -> int:
        if value == 0:
            raise ValueError("occurrence cannot be 0")
        return value


class ReplaceLinesOp(BaseModel):
    kind: Literal["replace_lines"] = "replace_lines"
    start_line: int = Field(
        description=(
            "The first line in the inclusive line range to replace. Positive values count from "
            "the beginning; negative values count backward from the end."
        )
    )
    end_line: int = Field(
        description=(
            "The last line in the inclusive line range to replace. Positive values count from "
            "the beginning; negative values count backward from the end."
        )
    )
    content: str = Field(description="The replacement content for the selected line range.")

    @field_validator("start_line", "end_line")
    @classmethod
    def validate_line_numbers(cls, value: int, info: ValidationInfo) -> int:
        if value == 0:
            raise ValueError(f"{info.field_name} cannot be 0")
        return value


class PatchOp(BaseModel):
    kind: Literal["patch"] = "patch"
    patch: str = Field(
        description=(
            "A unified diff patch or hunk-only patch to apply to the target file. The patch "
            "must apply cleanly to the current file content."
        )
    )

    @field_validator("patch")
    @classmethod
    def validate_patch(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("patch cannot be empty")
        return value


EditOperation = Annotated[
    ReplaceOp
    | AppendOp
    | PrependOp
    | DeleteOp
    | InsertBeforeOp
    | InsertAfterOp
    | ReplaceLinesOp
    | PatchOp,
    Field(discriminator="kind"),
]


class EditParams(BaseModel):
    path: str = Field(
        description=(
            "The path to the file to edit. Absolute paths are required when editing files "
            "outside the working directory."
        )
    )
    edit: list[EditOperation] = Field(
        description=(
            "The edit operation(s) to apply to the file. You can provide a single operation or "
            "a list of operations here. Supported kinds are `replace`, `append`, `prepend`, "
            "`delete`, `insert_before`, `insert_after`, `replace_lines`, and `patch`."
        ),
        min_length=1,
    )

    @field_validator("edit", mode="before")
    @classmethod
    def wrap_single_edit(cls, value: Any) -> Any:
        """Accept a single operation dict and wrap it into a list."""
        if isinstance(value, dict):
            return [value]
        if isinstance(value, BaseModel):
            return [value]
        return value


class _EditError(ValueError):
    pass


@dataclass(slots=True)
class _PatchLine:
    prefix: str
    text: str


@dataclass(slots=True)
class _PatchHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[_PatchLine]


class _BaseStructuredEditTool(CallableTool2[EditParams]):
    params: type[EditParams] = EditParams

    def __init__(self, runtime: Runtime, approval: Approval) -> None:
        super().__init__()
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs
        self._approval = approval
        self._plan_mode_checker: Callable[[], bool] | None = None
        self._plan_file_path_getter: Callable[[], Path | None] | None = None

    async def _validate_path(self, path: KaosPath) -> ToolError | None:
        """Validate that the path is safe to edit."""
        resolved_path = path.canonical()

        if (
            not is_within_workspace(resolved_path, self._work_dir, self._additional_dirs)
            and not path.is_absolute()
        ):
            return ToolError(
                message=(
                    f"`{path}` is not an absolute path. "
                    "You must provide an absolute path to edit a file "
                    "outside the working directory."
                ),
                brief="Invalid path",
            )
        return None

    @staticmethod
    def _find_occurrences(content: str, needle: str) -> list[int]:
        positions: list[int] = []
        start = 0
        while True:
            idx = content.find(needle, start)
            if idx == -1:
                return positions
            positions.append(idx)
            start = idx + len(needle)

    @staticmethod
    def _resolve_occurrence(positions: list[int], occurrence: int, *, label: str) -> int:
        if not positions:
            raise _EditError(f"{label} was not found in the file.")
        index = occurrence - 1 if occurrence > 0 else len(positions) + occurrence
        if index < 0 or index >= len(positions):
            raise _EditError(
                f"{label} occurrence {occurrence} is out of range; "
                f"found {len(positions)} match(es)."
            )
        return positions[index]

    @staticmethod
    def _normalize_line_number(value: int, total_lines: int, *, field_name: str) -> int:
        if total_lines == 0:
            raise _EditError("Cannot use replace_lines on an empty file.")
        normalized = value if value > 0 else total_lines + value + 1
        if normalized < 1 or normalized > total_lines:
            raise _EditError(
                f"{field_name} {value} is out of range for a file with {total_lines} line(s)."
            )
        return normalized

    @classmethod
    def _parse_patch_hunks(cls, patch_text: str) -> list[_PatchHunk]:
        raw_lines = patch_text.splitlines(keepends=True)
        hunks: list[_PatchHunk] = []
        i = 0
        while i < len(raw_lines):
            line = raw_lines[i]
            match = _HUNK_RE.match(line)
            if match is None:
                i += 1
                continue

            old_start = int(match.group(1))
            old_count = int(match.group(2) or "1")
            new_start = int(match.group(3))
            new_count = int(match.group(4) or "1")
            i += 1

            patch_lines: list[_PatchLine] = []
            while i < len(raw_lines):
                current = raw_lines[i]
                if _HUNK_RE.match(current):
                    break
                if current.startswith("\\ No newline at end of file"):
                    if not patch_lines:
                        raise _EditError(
                            "Patch marker 'No newline at end of file' has no preceding patch line."
                        )
                    previous = patch_lines[-1]
                    if previous.text.endswith("\n"):
                        previous.text = previous.text[:-1]
                    i += 1
                    continue
                if current[:1] in {" ", "+", "-"}:
                    patch_lines.append(_PatchLine(prefix=current[0], text=current[1:]))
                    i += 1
                    continue
                raise _EditError(f"Unsupported patch line: {current.rstrip()}.")

            hunks.append(
                _PatchHunk(
                    old_start=old_start,
                    old_count=old_count,
                    new_start=new_start,
                    new_count=new_count,
                    lines=patch_lines,
                )
            )

        if not hunks:
            raise _EditError("Patch does not contain any hunks.")
        return hunks

    @classmethod
    def _apply_patch(cls, content: str, op: PatchOp) -> str:
        original_lines = content.splitlines(keepends=True)
        result_lines: list[str] = []
        source_index = 0

        for hunk_number, hunk in enumerate(cls._parse_patch_hunks(op.patch), start=1):
            hunk_start = max(hunk.old_start - 1, 0)
            if hunk_start < source_index:
                raise _EditError(f"Patch hunk {hunk_number} is out of order or overlaps.")

            result_lines.extend(original_lines[source_index:hunk_start])
            source_index = hunk_start
            old_consumed = 0
            new_produced = 0

            for patch_line in hunk.lines:
                if patch_line.prefix == " ":
                    if (
                        source_index >= len(original_lines)
                        or original_lines[source_index] != patch_line.text
                    ):
                        raise _EditError(
                            f"Patch hunk {hunk_number} context did not match the file."
                        )
                    result_lines.append(original_lines[source_index])
                    source_index += 1
                    old_consumed += 1
                    new_produced += 1
                elif patch_line.prefix == "-":
                    if (
                        source_index >= len(original_lines)
                        or original_lines[source_index] != patch_line.text
                    ):
                        raise _EditError(
                            f"Patch hunk {hunk_number} deletion did not match the file."
                        )
                    source_index += 1
                    old_consumed += 1
                else:
                    result_lines.append(patch_line.text)
                    new_produced += 1

            if old_consumed != hunk.old_count:
                raise _EditError(
                    f"Patch hunk {hunk_number} expected {hunk.old_count} old line(s) "
                    f"but used {old_consumed}."
                )
            if new_produced != hunk.new_count:
                raise _EditError(
                    f"Patch hunk {hunk_number} expected {hunk.new_count} new line(s) "
                    f"but produced {new_produced}."
                )

        result_lines.extend(original_lines[source_index:])
        return "".join(result_lines)

    @staticmethod
    def _fuzzy_match_context(content: str, needle: str) -> str:
        """Find the closest fuzzy match for *needle* in *content* and return a
        diff-like context string showing where the closest match lives."""
        content_lines = content.splitlines(keepends=True)
        needle_lines = needle.splitlines(keepends=True)

        best_ratio = 0.0
        best_start = 0
        best_size = len(needle_lines)
        window = max(len(needle_lines), 1)

        for start in range(max(len(content_lines) - window + 1, 1)):
            candidate = content_lines[start : start + window]
            ratio = difflib.SequenceMatcher(None, "".join(candidate), needle).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_start = start
                best_size = window

        if best_ratio < 0.4:
            return ""

        # Show surrounding context (3 lines before/after)
        ctx_before = max(best_start - 3, 0)
        ctx_after = min(best_start + best_size + 3, len(content_lines))
        context_block = content_lines[ctx_before:ctx_after]

        closest_text = "".join(content_lines[best_start : best_start + best_size])
        diff = difflib.unified_diff(
            needle.splitlines(keepends=True),
            closest_text.splitlines(keepends=True),
            fromfile="expected",
            tofile="found (closest match)",
            lineterm="",
        )
        diff_str = "\n".join(diff)

        lines_info: list[str] = []
        lines_info.append(
            f"\nClosest match (similarity {best_ratio:.0%}) near line {best_start + 1}:"
        )
        if diff_str:
            lines_info.append(diff_str)
        lines_info.append(f"\nSurrounding context (lines {ctx_before + 1}-{ctx_after}):")
        for i, line in enumerate(context_block, start=ctx_before + 1):
            lines_info.append(f"  {i:>4} | {line.rstrip()}")
        return "\n".join(lines_info)

    def _apply_operation(self, content: str, op: EditOperation, *, index: int) -> str:
        if isinstance(op, ReplaceOp):
            if op.old not in content:
                hint = self._fuzzy_match_context(content, op.old)
                raise _EditError(
                    "No replacements were made. "
                    f"Replace operation {index} could not find the target string."
                    + (hint if hint else "")
                )
            if op.replace_all:
                return content.replace(op.old, op.new)
            return content.replace(op.old, op.new, 1)

        if isinstance(op, AppendOp):
            return content + op.content

        if isinstance(op, PrependOp):
            return op.content + content

        if isinstance(op, DeleteOp):
            if op.old not in content:
                hint = self._fuzzy_match_context(content, op.old)
                raise _EditError(
                    f"Delete operation {index} could not find the target string."
                    + (hint if hint else "")
                )
            if op.replace_all:
                return content.replace(op.old, "")
            return content.replace(op.old, "", 1)

        if isinstance(op, InsertBeforeOp):
            positions = self._find_occurrences(content, op.anchor)
            insert_at = self._resolve_occurrence(positions, op.occurrence, label="Anchor")
            return content[:insert_at] + op.content + content[insert_at:]

        if isinstance(op, InsertAfterOp):
            positions = self._find_occurrences(content, op.anchor)
            anchor_at = self._resolve_occurrence(positions, op.occurrence, label="Anchor")
            insert_at = anchor_at + len(op.anchor)
            return content[:insert_at] + op.content + content[insert_at:]

        if isinstance(op, ReplaceLinesOp):
            lines = content.splitlines(keepends=True)
            start_line = self._normalize_line_number(
                op.start_line,
                len(lines),
                field_name="start_line",
            )
            end_line = self._normalize_line_number(
                op.end_line,
                len(lines),
                field_name="end_line",
            )
            if start_line > end_line:
                raise _EditError(
                    f"Replace_lines operation {index} has start_line greater than end_line."
                )
            return "".join(lines[: start_line - 1]) + op.content + "".join(lines[end_line:])

        return self._apply_patch(content, op)

    @override
    async def __call__(self, params: EditParams) -> ToolReturnValue:
        if not params.path:
            return ToolError(
                message="File path cannot be empty.",
                brief="Empty file path",
            )

        try:
            p = KaosPath(params.path).expanduser()
            if err := await self._validate_path(p):
                return err
            p = p.canonical()

            plan_target = inspect_plan_edit_target(
                p,
                plan_mode_checker=self._plan_mode_checker,
                plan_file_path_getter=self._plan_file_path_getter,
            )
            if isinstance(plan_target, ToolError):
                return plan_target

            is_plan_file_edit = plan_target.is_plan_target

            if not await p.exists():
                if is_plan_file_edit:
                    return ToolError(
                        message=(
                            "The current plan file does not exist yet. "
                            "Use WriteFile to create it before calling Edit."
                        ),
                        brief="Plan file not created",
                    )
                return ToolError(
                    message=f"`{params.path}` does not exist.",
                    brief="File not found",
                )
            if not await p.is_file():
                return ToolError(
                    message=f"`{params.path}` is not a file.",
                    brief="Invalid path",
                )

            content = await p.read_text(errors="replace")
            original_content = content
            operations = params.edit

            for index, operation in enumerate(operations, start=1):
                updated_content = self._apply_operation(content, operation, index=index)
                if updated_content == content:
                    raise _EditError(f"Edit operation {index} made no changes.")
                content = updated_content

            diff_blocks: list[DisplayBlock] = await build_diff_blocks(
                str(p), original_content, content
            )

            action = (
                FileActions.EDIT
                if is_within_workspace(p, self._work_dir, self._additional_dirs)
                else FileActions.EDIT_OUTSIDE
            )

            # Plan file edits are auto-approved; all other edits need approval.
            if not is_plan_file_edit:
                result = await self._approval.request(
                    self.name,
                    action,
                    f"Edit file `{p}`",
                    display=diff_blocks,
                )
                if not result:
                    return result.rejection_error()

            await p.write_text(content, errors="replace")

            return ToolReturnValue(
                is_error=False,
                output="",
                message=(f"File successfully edited. Applied {len(operations)} edit operation(s)."),
                display=diff_blocks,
            )

        except _EditError as e:
            return ToolError(
                message=str(e),
                brief="Invalid edit",
            )
        except Exception as e:
            return ToolError(
                message=f"Failed to edit. Error: {e}",
                brief="Failed to edit file",
            )


class EditTool(_BaseStructuredEditTool):
    name: str = "Edit"
    description: str = load_desc(Path(__file__).parent / "edit.md")

    def __init__(self, runtime: Runtime, approval: Approval):
        super().__init__(runtime, approval)
        self._plan_mode_checker: Callable[[], bool] | None = None
        self._plan_file_path_getter: Callable[[], Path | None] | None = None

    def bind_plan_mode(
        self, checker: Callable[[], bool], path_getter: Callable[[], Path | None]
    ) -> None:
        self._plan_mode_checker = checker
        self._plan_file_path_getter = path_getter

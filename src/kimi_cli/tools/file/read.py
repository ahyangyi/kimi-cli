from collections import deque
from pathlib import Path
from typing import override

from kaos.path import KaosPath
from kosong.tooling import CallableTool2, ToolError, ToolOk, ToolReturnValue
from pydantic import BaseModel, Field, field_validator

from kimi_cli.soul.agent import Runtime
from kimi_cli.tools.file.utils import MEDIA_SNIFF_BYTES, detect_file_type
from kimi_cli.tools.utils import load_desc, truncate_line
from kimi_cli.utils.path import is_within_workspace
from kimi_cli.utils.sensitive import is_sensitive_file

MAX_LINES = 1000
MAX_LINE_LENGTH = 2000
MAX_BYTES = 100 << 10  # 100KB


class Params(BaseModel):
    path: str = Field(
        description=(
            "The path to the file to read. Absolute paths are required when reading files "
            "outside the working directory."
        )
    )
    line_offset: int = Field(
        description=(
            "The line number to start reading from. "
            "Positive values count from the beginning of the file. "
            "Negative values count backward from the end of the file, where -1 is the last "
            "line. By default read from the beginning of the file. "
            "Set this when the file is too large to read at once or when you want to read the "
            "tail of a file."
        ),
        default=1,
        json_schema_extra={"not": {"const": 0}},
    )
    n_lines: int = Field(
        description=(
            "The number of lines to read. "
            f"By default read up to {MAX_LINES} lines, which is the max allowed value. "
            "Set this value when the file is too large to read at once."
        ),
        default=MAX_LINES,
        ge=1,
    )

    @field_validator("line_offset")
    @classmethod
    def validate_line_offset(cls, value: int) -> int:
        if value == 0:
            raise ValueError("line_offset cannot be 0")
        return value


class ReadFile(CallableTool2[Params]):
    name: str = "ReadFile"
    params: type[Params] = Params

    def __init__(self, runtime: Runtime) -> None:
        description = load_desc(
            Path(__file__).parent / "read.md",
            {
                "MAX_LINES": MAX_LINES,
                "MAX_LINE_LENGTH": MAX_LINE_LENGTH,
                "MAX_BYTES": MAX_BYTES,
            },
        )
        super().__init__(description=description)
        self._runtime = runtime
        self._work_dir = runtime.builtin_args.KIMI_WORK_DIR
        self._additional_dirs = runtime.additional_dirs

    async def _validate_path(self, path: KaosPath) -> ToolError | None:
        """Validate that the path is safe to read."""
        resolved_path = path.canonical()

        if (
            not is_within_workspace(resolved_path, self._work_dir, self._additional_dirs)
            and not path.is_absolute()
        ):
            return ToolError(
                message=(
                    f"`{path}` is not an absolute path. "
                    "You must provide an absolute path to read a file "
                    "outside the working directory."
                ),
                brief="Invalid path",
            )
        return None

    @staticmethod
    def _build_message(
        *,
        lines_read: int,
        effective_line_offset: int,
        total_lines: int | None,
        stop_reason: str | None,
        truncated_line_numbers: list[int],
    ) -> str:
        if total_lines is None:
            if lines_read > 0:
                message = (
                    f"{lines_read} lines read from file starting from line {effective_line_offset}."
                )
            else:
                message = (
                    f"No lines read from file starting from line {effective_line_offset}."
                    if effective_line_offset != 1
                    else "No lines read from file."
                )
        else:
            message = (
                f"{lines_read} lines read from file starting from line {effective_line_offset}. "
                f"File has {total_lines} total lines."
                if lines_read > 0
                else (
                    f"No lines read from file starting from line {effective_line_offset}. "
                    f"File has {total_lines} total lines."
                    if effective_line_offset != 1 or total_lines != 0
                    else f"No lines read from file. File has {total_lines} total lines."
                )
            )

        if stop_reason == "max_lines":
            message += f" Max {MAX_LINES} lines reached."
        elif stop_reason == "max_bytes":
            message += f" Max {MAX_BYTES} bytes reached."
        elif stop_reason == "eof":
            message += " End of file reached."

        if truncated_line_numbers:
            message += f" Lines {truncated_line_numbers} were truncated."
        return message

    @staticmethod
    def _format_lines(lines: list[str], *, start_line: int) -> str:
        lines_with_no: list[str] = []
        for line_num, line in zip(range(start_line, start_line + len(lines)), lines, strict=True):
            lines_with_no.append(f"{line_num:6d}\t{line}")
        return "".join(lines_with_no)

    @override
    async def __call__(self, params: Params) -> ToolReturnValue:
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

            if is_sensitive_file(str(p)):
                return ToolError(
                    message=(
                        f"`{params.path}` appears to contain secrets "
                        "(matched sensitive file pattern). "
                        "Reading this file is blocked to protect credentials."
                    ),
                    brief="Sensitive file",
                )

            if not await p.exists():
                return ToolError(
                    message=f"`{params.path}` does not exist.",
                    brief="File not found",
                )
            if not await p.is_file():
                return ToolError(
                    message=f"`{params.path}` is not a file.",
                    brief="Invalid path",
                )

            header = await p.read_bytes(MEDIA_SNIFF_BYTES)
            file_type = detect_file_type(str(p), header=header)
            if file_type.kind in ("image", "video"):
                return ToolError(
                    message=(
                        f"`{params.path}` is a {file_type.kind} file. "
                        "Use other appropriate tools to read image or video files."
                    ),
                    brief="Unsupported file type",
                )

            if file_type.kind == "unknown":
                return ToolError(
                    message=(
                        f"`{params.path}` seems not readable. "
                        "You may need to read it with proper shell commands, Python tools "
                        "or MCP tools if available. "
                        "If you read/operate it with Python, you MUST ensure that any "
                        "third-party packages are installed in a virtual environment (venv)."
                    ),
                    brief="File not readable",
                )

            requested_line_limit = min(params.n_lines, MAX_LINES)
            lines: list[str] = []
            truncated_line_numbers: list[int] = []
            total_lines: int | None = None
            stop_reason: str | None = None
            n_bytes = 0

            if params.line_offset > 0:
                effective_line_offset = params.line_offset
                current_line_no = 0
                line_iter = p.read_lines(errors="replace").__aiter__()

                while True:
                    try:
                        line = await anext(line_iter)
                    except StopAsyncIteration:
                        total_lines = current_line_no
                        stop_reason = "eof"
                        break

                    current_line_no += 1
                    if current_line_no < effective_line_offset:
                        continue

                    truncated = truncate_line(line, MAX_LINE_LENGTH)
                    if truncated != line:
                        truncated_line_numbers.append(current_line_no)
                    lines.append(truncated)
                    n_bytes += len(truncated.encode("utf-8"))

                    if len(lines) >= requested_line_limit:
                        try:
                            await anext(line_iter)
                        except StopAsyncIteration:
                            total_lines = current_line_no
                            stop_reason = "eof"
                        else:
                            if params.n_lines > MAX_LINES:
                                stop_reason = "max_lines"
                        break

                    if n_bytes >= MAX_BYTES:
                        stop_reason = "max_bytes"
                        break
            else:
                effective_line_offset = 1
                total_lines = 0
                tail_lines: deque[tuple[int, str]] = deque(maxlen=max(1, -params.line_offset))

                async for line in p.read_lines(errors="replace"):
                    total_lines += 1
                    tail_lines.append((total_lines, line))

                effective_line_offset = max(1, total_lines + params.line_offset + 1)
                available_from_offset = max(0, total_lines - effective_line_offset + 1)

                for current_line_no, line in tail_lines:
                    if current_line_no < effective_line_offset:
                        continue

                    truncated = truncate_line(line, MAX_LINE_LENGTH)
                    if truncated != line:
                        truncated_line_numbers.append(current_line_no)
                    lines.append(truncated)
                    n_bytes += len(truncated.encode("utf-8"))

                    if len(lines) >= requested_line_limit:
                        if params.n_lines > MAX_LINES and available_from_offset > MAX_LINES:
                            stop_reason = "max_lines"
                        break

                    if n_bytes >= MAX_BYTES:
                        stop_reason = "max_bytes"
                        break

                if stop_reason is None and available_from_offset <= len(lines):
                    stop_reason = "eof"

            return ToolOk(
                output=self._format_lines(lines, start_line=effective_line_offset),
                message=self._build_message(
                    lines_read=len(lines),
                    effective_line_offset=effective_line_offset,
                    total_lines=total_lines,
                    stop_reason=stop_reason,
                    truncated_line_numbers=truncated_line_numbers,
                ),
            )
        except Exception as e:
            return ToolError(
                message=f"Failed to read {params.path}. Error: {e}",
                brief="Failed to read file",
            )

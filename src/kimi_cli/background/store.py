from __future__ import annotations

import contextlib
import os
import re
import tempfile
from array import array
from pathlib import Path

from pydantic import BaseModel, ValidationError

from kimi_cli.utils.io import atomic_json_write
from kimi_cli.utils.logging import logger

from .models import (
    TaskConsumerState,
    TaskControl,
    TaskOutputChunk,
    TaskOutputLineChunk,
    TaskRuntime,
    TaskSpec,
    TaskStatus,
    TaskView,
)

_VALID_TASK_ID = re.compile(r"^[a-z0-9][a-z0-9\-]{1,24}$")
_SCAN_CHUNK = 1 << 16  # 64 KiB – chunk size for binary scanning


# ---------------------------------------------------------------------------
# LineIndex – sidecar binary index mapping line numbers to byte offsets
# ---------------------------------------------------------------------------


class LineIndex:
    """Incremental line index for an append-only log file.

    Persisted as a binary sidecar (``<log>.lidx``) so that repeated reads
    only scan *new* content.  Both forward and tail reads become O(1) seek +
    O(window) I/O once the index is up to date.

    **Sidecar format** (native byte-order ``uint64`` array):

    * ``data[0]`` – *scanned_to*: byte position up to which the log was indexed.
    * ``data[1]`` – *flags*: bit 0 = ``last_was_newline``.
    * ``data[2..]`` – packed line-start byte offsets (one per line).
    """

    INDEX_SUFFIX = ".lidx"

    def __init__(self, log_path: Path) -> None:
        self._log_path = log_path
        self._idx_path = log_path.parent / (log_path.name + self.INDEX_SUFFIX)
        self._offsets: array[int] = array("Q")
        self._scanned_to: int = 0
        self._last_was_newline: bool = False
        # None = not yet attempted, True = loaded OK, False = attempted & failed
        self._sidecar_load_ok: bool | None = None

    # -- persistence --------------------------------------------------------

    def _load_sidecar(self) -> bool:
        """Load index from sidecar file.  Returns ``True`` on success.

        The result is cached: subsequent calls return the same value without
        re-reading the sidecar, so a failed load is never silently reported
        as success.
        """
        if self._sidecar_load_ok is not None:
            return self._sidecar_load_ok
        if not self._idx_path.exists():
            self._sidecar_load_ok = False
            return False
        try:
            raw = self._idx_path.read_bytes()
            if len(raw) < 16 or len(raw) % 8 != 0:
                self._sidecar_load_ok = False
                return False
            data: array[int] = array("Q")
            data.frombytes(raw)
            scanned_to = data[0]
            flags = data[1]
            offsets: array[int] = array("Q")
            if len(data) > 2:
                offsets.extend(data[2:])
            # Basic validation: offsets must be strictly increasing and
            # within the scanned region.
            for i in range(1, len(offsets)):
                if offsets[i] <= offsets[i - 1]:
                    self._sidecar_load_ok = False
                    return False
            if offsets and offsets[-1] >= scanned_to:
                self._sidecar_load_ok = False
                return False
            self._scanned_to = scanned_to
            self._last_was_newline = bool(flags & 1)
            self._offsets = offsets
            self._sidecar_load_ok = True
            return True
        except Exception:
            self._sidecar_load_ok = False
            return False

    def _save_sidecar(self) -> None:
        """Persist index to sidecar file (atomic via tmp + rename).

        Uses :func:`os.fdopen` so the file descriptor has a single clear
        owner (the ``with`` block) — no manual double-close handling needed.
        """
        data: array[int] = array("Q")
        data.append(self._scanned_to)
        data.append(1 if self._last_was_newline else 0)
        data.extend(self._offsets)
        try:
            fd, tmp_path = tempfile.mkstemp(
                dir=self._idx_path.parent,
                suffix=".lidx.tmp",
            )
            try:
                with os.fdopen(fd, "wb") as f:
                    f.write(data.tobytes())
                    f.flush()
                    os.fsync(f.fileno())
                # fd is now closed by the context manager; rename to final path.
                os.replace(tmp_path, self._idx_path)
            except BaseException:
                # fd already closed by `with` (or never opened if fdopen failed).
                with contextlib.suppress(OSError):
                    os.unlink(tmp_path)
        except OSError:
            pass  # best-effort; the index can always be rebuilt

    # -- public API ---------------------------------------------------------

    def refresh(self) -> None:
        """Update the index to reflect the current log file on disk."""
        if not self._log_path.exists():
            self._offsets = array("Q")
            self._scanned_to = 0
            self._last_was_newline = False
            return

        file_size = self._log_path.stat().st_size
        if file_size == 0:
            self._offsets = array("Q")
            self._scanned_to = 0
            self._last_was_newline = False
            return

        # Lazy-load the sidecar on first access.
        if self._scanned_to == 0 and len(self._offsets) == 0:
            self._load_sidecar()

        # File truncated → rebuild.
        if file_size < self._scanned_to:
            self._offsets = array("Q")
            self._scanned_to = 0
            self._last_was_newline = False

        if file_size == self._scanned_to:
            return  # already up to date

        # -- incremental scan -----------------------------------------------
        with self._log_path.open("rb") as f:
            if self._scanned_to == 0:
                # First byte always starts line 0.
                self._offsets.append(0)
                self._last_was_newline = False
            elif self._last_was_newline:
                # Previous scan ended on '\n'; new content starts a fresh line.
                self._offsets.append(self._scanned_to)
                self._last_was_newline = False

            f.seek(self._scanned_to)
            while True:
                chunk = f.read(_SCAN_CHUNK)
                if not chunk:
                    break
                base = f.tell() - len(chunk)
                pos = 0
                while True:
                    idx = chunk.find(b"\n", pos)
                    if idx < 0:
                        break
                    next_byte = base + idx + 1
                    if next_byte < file_size:
                        self._offsets.append(next_byte)
                    pos = idx + 1
                self._last_was_newline = chunk[-1:] == b"\n"

        self._scanned_to = file_size
        self._save_sidecar()

    @property
    def line_count(self) -> int:
        """Number of indexed lines."""
        return len(self._offsets)

    def line_byte_offset(self, line: int) -> int:
        """Byte offset where *line* starts (0-based)."""
        return self._offsets[line]

    def line_end_byte_offset(self, line: int) -> int:
        """Byte offset where *line* ends (exclusive)."""
        if line + 1 < len(self._offsets):
            return self._offsets[line + 1]
        return self._scanned_to


# ---------------------------------------------------------------------------


def _validate_task_id(task_id: str) -> None:
    if not _VALID_TASK_ID.match(task_id):
        raise ValueError(f"Invalid task_id: {task_id!r}")


class BackgroundTaskStore:
    SPEC_FILE = "spec.json"
    RUNTIME_FILE = "runtime.json"
    CONTROL_FILE = "control.json"
    CONSUMER_FILE = "consumer.json"
    OUTPUT_FILE = "output.log"

    def __init__(self, root: Path):
        self._root = root

    @property
    def root(self) -> Path:
        return self._root

    def _ensure_root(self) -> Path:
        """Return the root directory, creating it if it does not exist."""
        self._root.mkdir(parents=True, exist_ok=True)
        return self._root

    def task_dir(self, task_id: str) -> Path:
        _validate_task_id(task_id)
        path = self._ensure_root() / task_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def task_path(self, task_id: str) -> Path:
        _validate_task_id(task_id)
        return self.root / task_id

    def spec_path(self, task_id: str) -> Path:
        return self.task_path(task_id) / self.SPEC_FILE

    def runtime_path(self, task_id: str) -> Path:
        return self.task_path(task_id) / self.RUNTIME_FILE

    def control_path(self, task_id: str) -> Path:
        return self.task_path(task_id) / self.CONTROL_FILE

    def consumer_path(self, task_id: str) -> Path:
        return self.task_path(task_id) / self.CONSUMER_FILE

    def output_path(self, task_id: str) -> Path:
        return self.task_path(task_id) / self.OUTPUT_FILE

    def create_task(self, spec: TaskSpec) -> None:
        task_dir = self.task_dir(spec.id)
        atomic_json_write(spec.model_dump(mode="json"), task_dir / self.SPEC_FILE)
        atomic_json_write(TaskRuntime().model_dump(mode="json"), task_dir / self.RUNTIME_FILE)
        atomic_json_write(TaskControl().model_dump(mode="json"), task_dir / self.CONTROL_FILE)
        atomic_json_write(
            TaskConsumerState().model_dump(mode="json"),
            task_dir / self.CONSUMER_FILE,
        )
        self.output_path(spec.id).touch(exist_ok=True)

    def list_task_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        task_ids: list[str] = []
        for path in sorted(self.root.iterdir()):
            if not path.is_dir():
                continue
            if not (path / self.SPEC_FILE).exists():
                continue
            task_ids.append(path.name)
        return task_ids

    def count_active_runtimes(self, *, max_count: int | None = None) -> int:
        """Count tasks with non-terminal runtime status, reading only runtime.json.

        Stops early when *max_count* is reached (useful for limit checks).
        """
        from .models import TERMINAL_TASK_STATUSES

        if not self.root.exists():
            return 0
        count = 0
        for path in self.root.iterdir():
            if not path.is_dir():
                continue
            runtime_path = path / self.RUNTIME_FILE
            if not runtime_path.exists():
                continue
            try:
                runtime = TaskRuntime.model_validate_json(runtime_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if runtime.status not in TERMINAL_TASK_STATUSES:
                count += 1
                if max_count is not None and count >= max_count:
                    return count
        return count

    def write_spec(self, spec: TaskSpec) -> None:
        atomic_json_write(spec.model_dump(mode="json"), self.spec_path(spec.id))

    def read_spec(self, task_id: str) -> TaskSpec:
        return TaskSpec.model_validate_json(self.spec_path(task_id).read_text(encoding="utf-8"))

    def write_runtime(self, task_id: str, runtime: TaskRuntime) -> None:
        atomic_json_write(runtime.model_dump(mode="json"), self.runtime_path(task_id))

    def read_runtime(self, task_id: str) -> TaskRuntime:
        path = self.runtime_path(task_id)
        if not path.exists():
            return TaskRuntime()
        return _read_json_model(
            path,
            TaskRuntime,
            fallback=TaskRuntime(updated_at=0),
            artifact="task runtime",
        )

    def write_control(self, task_id: str, control: TaskControl) -> None:
        atomic_json_write(control.model_dump(mode="json"), self.control_path(task_id))

    def read_control(self, task_id: str) -> TaskControl:
        path = self.control_path(task_id)
        if not path.exists():
            return TaskControl()
        return _read_json_model(
            path,
            TaskControl,
            fallback=TaskControl(),
            artifact="task control",
        )

    def write_consumer(self, task_id: str, consumer: TaskConsumerState) -> None:
        atomic_json_write(consumer.model_dump(mode="json"), self.consumer_path(task_id))

    def read_consumer(self, task_id: str) -> TaskConsumerState:
        path = self.consumer_path(task_id)
        if not path.exists():
            return TaskConsumerState()
        return _read_json_model(
            path,
            TaskConsumerState,
            fallback=TaskConsumerState(),
            artifact="task consumer state",
        )

    def merged_view(self, task_id: str) -> TaskView:
        return TaskView(
            spec=self.read_spec(task_id),
            runtime=self.read_runtime(task_id),
            control=self.read_control(task_id),
            consumer=self.read_consumer(task_id),
        )

    def list_views(self) -> list[TaskView]:
        views: list[TaskView] = []
        for task_id in self.list_task_ids():
            try:
                views.append(self.merged_view(task_id))
            except (OSError, ValidationError, ValueError, UnicodeDecodeError) as exc:
                logger.warning(
                    "Skipping invalid background task {task_id} from {path}: {error}",
                    task_id=task_id,
                    path=self.root / task_id / self.SPEC_FILE,
                    error=exc,
                )
        views.sort(
            key=lambda view: view.runtime.updated_at or view.spec.created_at,
            reverse=True,
        )
        return views

    # -- byte-offset output reading (legacy) ----------------------------------

    def read_output(
        self,
        task_id: str,
        offset: int,
        max_bytes: int,
        *,
        status: TaskStatus,
        path_override: Path | None = None,
    ) -> TaskOutputChunk:
        path = path_override if path_override is not None else self.output_path(task_id)
        if not path.exists():
            return TaskOutputChunk(
                task_id=task_id,
                offset=offset,
                next_offset=offset,
                text="",
                eof=True,
                status=status,
            )

        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            total_size = f.tell()
            bounded_offset = min(max(offset, 0), total_size)
            f.seek(bounded_offset)
            content = f.read(max_bytes)

        next_offset = bounded_offset + len(content)
        return TaskOutputChunk(
            task_id=task_id,
            offset=bounded_offset,
            next_offset=next_offset,
            text=content.decode("utf-8", errors="replace"),
            eof=next_offset >= total_size,
            status=status,
        )

    def tail_output(self, task_id: str, max_bytes: int, max_lines: int) -> str:
        path = self.output_path(task_id)
        if not path.exists():
            return ""

        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            total_size = f.tell()
            start = max(0, total_size - max_bytes)
            f.seek(start)
            content = f.read()

        text = content.decode("utf-8", errors="replace")
        lines = text.splitlines()
        if len(lines) > max_lines:
            lines = lines[-max_lines:]
        return "\n".join(lines)

    # -- line-index output reading (new) --------------------------------------

    def read_output_lines(
        self,
        task_id: str,
        line_offset: int | None,
        max_bytes: int,
        *,
        status: TaskStatus,
    ) -> TaskOutputLineChunk:
        """Read output by lines, capping the returned payload at *max_bytes*.

        *line_offset*: 0-based line number to start reading from.  When ``None``
        the tail (last lines fitting within *max_bytes*) is returned.

        Uses the :class:`LineIndex` sidecar so that both forward and tail reads
        are O(1)-seek + O(window) I/O after the index is up to date.
        """
        path = self.output_path(task_id)
        output_path_str = str(path.resolve())
        empty = TaskOutputLineChunk(
            task_id=task_id,
            start_line=0,
            end_line=0,
            has_before=False,
            has_after=False,
            next_offset=None,
            text="",
            status=status,
            output_path=output_path_str,
            line_too_large=False,
        )
        if not path.exists():
            return empty

        index = LineIndex(path)
        index.refresh()

        if index.line_count == 0:
            return empty

        if line_offset is not None:
            return self._read_forward(
                path,
                index,
                line_offset,
                max_bytes,
                task_id,
                status,
                output_path_str,
            )
        return self._read_tail(
            path,
            index,
            max_bytes,
            task_id,
            status,
            output_path_str,
        )

    # -- helpers for read_output_lines -----------------------------------------

    def _read_forward(
        self,
        path: Path,
        index: LineIndex,
        line_offset: int,
        max_bytes: int,
        task_id: str,
        status: TaskStatus,
        output_path_str: str,
    ) -> TaskOutputLineChunk:
        """Forward read starting at *line_offset* using the line index."""
        n = index.line_count

        if line_offset >= n:
            capped = n
            return TaskOutputLineChunk(
                task_id=task_id,
                start_line=capped,
                end_line=capped,
                has_before=capped > 0,
                has_after=False,
                next_offset=None,
                text="",
                status=status,
                output_path=output_path_str,
                line_too_large=False,
            )

        start_line = line_offset
        end_line = line_offset
        total_bytes = 0

        while end_line < n:
            line_size = index.line_end_byte_offset(end_line) - index.line_byte_offset(end_line)
            if end_line == start_line and line_size > max_bytes:
                # First candidate line exceeds budget on its own.
                has_after = (start_line + 1) < n
                return TaskOutputLineChunk(
                    task_id=task_id,
                    start_line=start_line,
                    end_line=start_line,
                    has_before=start_line > 0,
                    has_after=has_after,
                    next_offset=start_line + 1 if has_after else None,
                    text="",
                    status=status,
                    output_path=output_path_str,
                    line_too_large=True,
                )
            if total_bytes + line_size > max_bytes:
                break
            total_bytes += line_size
            end_line += 1

        start_byte = index.line_byte_offset(start_line)
        with path.open("rb") as f:
            f.seek(start_byte)
            raw = f.read(total_bytes)

        text = raw.decode("utf-8", errors="replace").rstrip("\n")
        has_after = end_line < n
        return TaskOutputLineChunk(
            task_id=task_id,
            start_line=start_line,
            end_line=end_line,
            has_before=start_line > 0,
            has_after=has_after,
            next_offset=end_line if has_after else None,
            text=text,
            status=status,
            output_path=output_path_str,
            line_too_large=False,
        )

    def _read_tail(
        self,
        path: Path,
        index: LineIndex,
        max_bytes: int,
        task_id: str,
        status: TaskStatus,
        output_path_str: str,
    ) -> TaskOutputLineChunk:
        """Tail read using the line index – walk backward to find the window."""
        n = index.line_count
        total_bytes = 0
        start_line = n

        for i in range(n - 1, -1, -1):
            line_size = index.line_end_byte_offset(i) - index.line_byte_offset(i)
            if total_bytes + line_size > max_bytes:
                if start_line == n:
                    # Cannot fit even the very last line.
                    return TaskOutputLineChunk(
                        task_id=task_id,
                        start_line=n - 1,
                        end_line=n - 1,
                        has_before=n - 1 > 0,
                        has_after=False,
                        next_offset=None,
                        text="",
                        status=status,
                        output_path=output_path_str,
                        line_too_large=True,
                    )
                break
            total_bytes += line_size
            start_line = i

        start_byte = index.line_byte_offset(start_line)
        with path.open("rb") as f:
            f.seek(start_byte)
            raw = f.read(total_bytes)

        text = raw.decode("utf-8", errors="replace").rstrip("\n")
        return TaskOutputLineChunk(
            task_id=task_id,
            start_line=start_line,
            end_line=n,
            has_before=start_line > 0,
            has_after=False,
            next_offset=None,
            text=text,
            status=status,
            output_path=output_path_str,
            line_too_large=False,
        )

    def prune(self, *, max_age_s: float = 7 * 24 * 3600) -> list[str]:
        """Remove task directories older than *max_age_s* seconds (default 7 days).

        Only terminal tasks are pruned.
        """
        import shutil
        import time

        from .models import TERMINAL_TASK_STATUSES

        if not self.root.exists():
            return []
        now = time.time()
        pruned: list[str] = []
        for path in list(self.root.iterdir()):
            if not path.is_dir():
                continue
            runtime_path = path / self.RUNTIME_FILE
            if not runtime_path.exists():
                continue
            try:
                runtime = TaskRuntime.model_validate_json(runtime_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if runtime.status not in TERMINAL_TASK_STATUSES:
                continue
            finished_at = runtime.finished_at or runtime.updated_at
            if now - finished_at >= max_age_s:
                shutil.rmtree(path, ignore_errors=True)
                pruned.append(path.name)
        return pruned


def _read_json_model[T: BaseModel](path: Path, model: type[T], *, fallback: T, artifact: str) -> T:
    try:
        return model.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError, UnicodeDecodeError) as exc:
        logger.warning(
            "Failed to read {artifact} from {path}; using defaults: {error}",
            artifact=artifact,
            path=path,
            error=exc,
        )
        return fallback

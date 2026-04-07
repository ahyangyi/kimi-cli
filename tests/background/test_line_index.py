"""Tests for the LineIndex sidecar line-offset index."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from kimi_cli.background.models import TaskRuntime, TaskSpec
from kimi_cli.background.store import BackgroundTaskStore, LineIndex


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _append(path: Path, content: str) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(content)


# ---------------------------------------------------------------------------
# Basic line counting
# ---------------------------------------------------------------------------


def test_empty_file(tmp_path: Path) -> None:
    p = tmp_path / "output.log"
    p.write_bytes(b"")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 0


def test_file_does_not_exist(tmp_path: Path) -> None:
    idx = LineIndex(tmp_path / "nonexistent.log")
    idx.refresh()
    assert idx.line_count == 0


def test_single_line_with_newline(tmp_path: Path) -> None:
    p = tmp_path / "output.log"
    _write(p, "hello\n")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 1
    assert idx.line_byte_offset(0) == 0
    assert idx.line_end_byte_offset(0) == 6  # len("hello\n")


def test_single_line_without_newline(tmp_path: Path) -> None:
    p = tmp_path / "output.log"
    _write(p, "hello")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 1
    assert idx.line_byte_offset(0) == 0
    assert idx.line_end_byte_offset(0) == 5


def test_multiple_lines_trailing_newline(tmp_path: Path) -> None:
    p = tmp_path / "output.log"
    _write(p, "a\nb\nc\n")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 3
    assert idx.line_byte_offset(0) == 0
    assert idx.line_byte_offset(1) == 2
    assert idx.line_byte_offset(2) == 4
    assert idx.line_end_byte_offset(2) == 6


def test_multiple_lines_no_trailing_newline(tmp_path: Path) -> None:
    p = tmp_path / "output.log"
    _write(p, "a\nb\nc")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 3
    assert idx.line_end_byte_offset(2) == 5  # scanned_to


def test_only_newline(tmp_path: Path) -> None:
    p = tmp_path / "output.log"
    _write(p, "\n")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 1


def test_empty_lines(tmp_path: Path) -> None:
    """Multiple consecutive newlines -> multiple lines."""
    p = tmp_path / "output.log"
    _write(p, "a\n\nb\n")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 3
    assert idx.line_byte_offset(1) == 2  # empty line between a and b


# ---------------------------------------------------------------------------
# Incremental indexing (file grows)
# ---------------------------------------------------------------------------


def test_incremental_append_with_trailing_newline(tmp_path: Path) -> None:
    """Index stays correct when the file grows and previous content ended with \\n."""
    p = tmp_path / "output.log"
    _write(p, "a\nb\n")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 2

    _append(p, "c\n")
    idx.refresh()
    assert idx.line_count == 3
    assert idx.line_byte_offset(2) == 4


def test_incremental_append_without_trailing_newline(tmp_path: Path) -> None:
    """File initially ends without \\n, then more content is appended."""
    p = tmp_path / "output.log"
    _write(p, "a\nb")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 2

    # Append completes line "b" and adds line "c"
    _append(p, "\nc\n")
    idx.refresh()
    assert idx.line_count == 3
    assert idx.line_byte_offset(2) == 4


def test_incremental_extends_unterminated_line(tmp_path: Path) -> None:
    """Appending content without \\n extends the last line, not a new one."""
    p = tmp_path / "output.log"
    _write(p, "hello")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 1

    _append(p, " world")
    idx.refresh()
    assert idx.line_count == 1
    assert idx.line_end_byte_offset(0) == len("hello world")


def test_incremental_multiple_stages(tmp_path: Path) -> None:
    """Simulate a process writing output incrementally."""
    p = tmp_path / "output.log"
    p.write_bytes(b"")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 0

    _append(p, "step 1\n")
    idx.refresh()
    assert idx.line_count == 1

    _append(p, "step 2\nstep 3\n")
    idx.refresh()
    assert idx.line_count == 3

    _append(p, "step 4")
    idx.refresh()
    assert idx.line_count == 4


# ---------------------------------------------------------------------------
# File truncation / corruption
# ---------------------------------------------------------------------------


def test_truncation_rebuilds_index(tmp_path: Path) -> None:
    p = tmp_path / "output.log"
    _write(p, "a\nb\nc\n")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 3

    _write(p, "x\n")  # truncate + rewrite
    idx.refresh()
    assert idx.line_count == 1
    assert idx.line_byte_offset(0) == 0


# ---------------------------------------------------------------------------
# Sidecar persistence
# ---------------------------------------------------------------------------


def test_sidecar_round_trip(tmp_path: Path) -> None:
    """Index saved to disk is reloaded correctly by a new LineIndex instance."""
    p = tmp_path / "output.log"
    _write(p, "alpha\nbeta\ngamma\n")
    idx1 = LineIndex(p)
    idx1.refresh()
    assert idx1.line_count == 3

    idx2 = LineIndex(p)
    idx2.refresh()  # should load from sidecar, no re-scan
    assert idx2.line_count == 3
    assert idx2.line_byte_offset(1) == idx1.line_byte_offset(1)
    assert idx2.line_byte_offset(2) == idx1.line_byte_offset(2)


def test_sidecar_incremental_after_reload(tmp_path: Path) -> None:
    """After reloading from sidecar, incremental scan picks up new lines."""
    p = tmp_path / "output.log"
    _write(p, "a\n")
    idx1 = LineIndex(p)
    idx1.refresh()
    assert idx1.line_count == 1

    _append(p, "b\n")
    idx2 = LineIndex(p)
    idx2.refresh()
    assert idx2.line_count == 2


def test_corrupted_sidecar_rebuilds(tmp_path: Path) -> None:
    """Corrupted sidecar file doesn't break the index; it rebuilds."""
    p = tmp_path / "output.log"
    _write(p, "a\nb\n")
    idx_path = p.parent / (p.name + LineIndex.INDEX_SUFFIX)
    idx_path.write_bytes(b"garbage")

    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 2


def test_sidecar_with_out_of_range_offsets_rebuilds(tmp_path: Path) -> None:
    """Sidecar whose offsets fall outside the scanned region is rejected."""
    from array import array as _array

    p = tmp_path / "output.log"
    _write(p, "a\nb\n")  # 4 bytes total
    idx_path = p.parent / (p.name + LineIndex.INDEX_SUFFIX)

    # Craft a sidecar where scanned_to=4 but last offset=10 (out of range).
    bad: _array[int] = _array("Q", [4, 0, 0, 10])
    idx_path.write_bytes(bad.tobytes())

    idx = LineIndex(p)
    idx.refresh()
    # Must have rebuilt from scratch, producing 2 correct lines.
    assert idx.line_count == 2
    assert idx.line_byte_offset(0) == 0
    assert idx.line_byte_offset(1) == 2


def test_sidecar_is_written_atomically(tmp_path: Path) -> None:
    """After refresh, the sidecar exists and no temp files are left behind."""
    p = tmp_path / "output.log"
    _write(p, "x\ny\nz\n")

    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 3

    idx_path = p.parent / (p.name + LineIndex.INDEX_SUFFIX)
    assert idx_path.exists(), "sidecar should have been written"
    # No leftover .lidx.tmp files
    tmp_files = list(p.parent.glob("*.lidx.tmp"))
    assert tmp_files == [], f"temp sidecar files should be cleaned up: {tmp_files}"


# ---------------------------------------------------------------------------
# Oversized / large files
# ---------------------------------------------------------------------------


def test_many_lines(tmp_path: Path) -> None:
    """Index handles hundreds of lines correctly."""
    p = tmp_path / "output.log"
    n = 500
    _write(p, "".join(f"line {i}\n" for i in range(n)))
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == n
    # Spot-check a few offsets
    assert idx.line_byte_offset(0) == 0
    # Last line
    last_start = idx.line_byte_offset(n - 1)
    last_end = idx.line_end_byte_offset(n - 1)
    expected_last = f"line {n - 1}\n"
    with p.open("rb") as f:
        f.seek(last_start)
        assert f.read(last_end - last_start) == expected_last.encode()


def test_oversized_single_line(tmp_path: Path) -> None:
    """A single very long line is indexed as exactly one line."""
    p = tmp_path / "output.log"
    _write(p, "x" * 200_000 + "\n")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 1
    assert idx.line_byte_offset(0) == 0
    assert idx.line_end_byte_offset(0) == 200_001


def test_multibyte_utf8(tmp_path: Path) -> None:
    """Line offsets are byte-based and handle multi-byte UTF-8 correctly."""
    p = tmp_path / "output.log"
    # each CJK char is 3 bytes in UTF-8
    _write(p, "\u4f60\u597d\n\u4e16\u754c\n")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 2
    assert idx.line_byte_offset(0) == 0
    assert idx.line_byte_offset(1) == len("\u4f60\u597d\n".encode())


# ---------------------------------------------------------------------------
# Idempotent refresh
# ---------------------------------------------------------------------------


def test_double_refresh_is_noop(tmp_path: Path) -> None:
    p = tmp_path / "output.log"
    _write(p, "a\nb\n")
    idx = LineIndex(p)
    idx.refresh()
    count1 = idx.line_count
    idx.refresh()
    assert idx.line_count == count1


# ---------------------------------------------------------------------------
# Store-level read_output_lines integration (forward & tail via LineIndex)
# ---------------------------------------------------------------------------


def _make_store_task(tmp_path: Path, task_id: str, output: str) -> BackgroundTaskStore:
    """Create a minimal store with one completed task."""
    store = BackgroundTaskStore(tmp_path / "tasks")
    spec = TaskSpec(
        id=task_id,
        kind="bash",
        session_id="s1",
        description="test",
        tool_call_id="tc1",
        command="echo hi",
        shell_name="bash",
        shell_path="/bin/bash",
        cwd="/tmp",
        timeout_s=60,
    )
    store.create_task(spec)
    store.output_path(task_id).write_text(output, encoding="utf-8")
    store.write_runtime(
        task_id,
        TaskRuntime(status="completed", exit_code=0, finished_at=time.time()),
    )
    return store


def test_store_forward_read_basic(tmp_path: Path) -> None:
    store = _make_store_task(tmp_path, "btest0001", "a\nb\nc\n")
    chunk = store.read_output_lines("btest0001", 0, 999_999, status="completed")
    assert chunk.start_line == 0
    assert chunk.end_line == 3
    assert chunk.has_before is False
    assert chunk.has_after is False
    assert chunk.next_offset is None
    assert chunk.text == "a\nb\nc"


def test_store_forward_read_with_offset(tmp_path: Path) -> None:
    store = _make_store_task(tmp_path, "btest0002", "a\nb\nc\nd\ne\n")
    chunk = store.read_output_lines("btest0002", 2, 999_999, status="completed")
    assert chunk.start_line == 2
    assert chunk.end_line == 5
    assert chunk.has_before is True
    assert chunk.has_after is False
    assert chunk.text == "c\nd\ne"


def test_store_forward_read_budget_limits_lines(tmp_path: Path) -> None:
    store = _make_store_task(tmp_path, "btest0003", "aaa\nbbb\nccc\n")
    # Budget of 4 bytes fits "aaa\n" (4 bytes) but not "aaa\nbbb\n" (8 bytes)
    chunk = store.read_output_lines("btest0003", 0, 4, status="completed")
    assert chunk.start_line == 0
    assert chunk.end_line == 1
    assert chunk.has_before is False
    assert chunk.has_after is True
    assert chunk.next_offset == 1
    assert chunk.text == "aaa"


def test_store_forward_read_oversized_first_line(tmp_path: Path) -> None:
    huge = "x" * 200 + "\nsmall\n"
    store = _make_store_task(tmp_path, "btest0004", huge)
    chunk = store.read_output_lines("btest0004", 0, 50, status="completed")
    assert chunk.line_too_large is True
    assert chunk.start_line == 0
    assert chunk.end_line == 0
    assert chunk.has_after is True
    assert chunk.next_offset == 1


def test_store_forward_read_offset_past_end(tmp_path: Path) -> None:
    store = _make_store_task(tmp_path, "btest0005", "a\n")
    chunk = store.read_output_lines("btest0005", 999, 999_999, status="completed")
    assert chunk.start_line == 1  # capped to line_count
    assert chunk.end_line == 1
    assert chunk.has_before is True
    assert chunk.has_after is False
    assert chunk.text == ""


def test_store_tail_read_basic(tmp_path: Path) -> None:
    store = _make_store_task(tmp_path, "btest0006", "a\nb\nc\n")
    chunk = store.read_output_lines("btest0006", None, 999_999, status="completed")
    assert chunk.start_line == 0
    assert chunk.end_line == 3
    assert chunk.has_before is False
    assert chunk.has_after is False
    assert chunk.text == "a\nb\nc"


def test_store_tail_read_budget_limits_lines(tmp_path: Path) -> None:
    store = _make_store_task(tmp_path, "btest0007", "aaa\nbbb\nccc\n")
    # Budget of 4 bytes fits only the last line "ccc\n" (4 bytes)
    chunk = store.read_output_lines("btest0007", None, 4, status="completed")
    assert chunk.start_line == 2
    assert chunk.end_line == 3
    assert chunk.has_before is True
    assert chunk.has_after is False
    assert chunk.text == "ccc"


def test_store_tail_read_oversized_single_line(tmp_path: Path) -> None:
    huge = "x" * 200 + "\n"
    store = _make_store_task(tmp_path, "btest0008", huge)
    chunk = store.read_output_lines("btest0008", None, 50, status="completed")
    assert chunk.line_too_large is True
    assert chunk.text == ""


def test_store_tail_no_trailing_newline(tmp_path: Path) -> None:
    store = _make_store_task(tmp_path, "btest0009", "alpha\nbeta")
    chunk = store.read_output_lines("btest0009", None, 999_999, status="completed")
    assert chunk.start_line == 0
    assert chunk.end_line == 2
    assert chunk.text == "alpha\nbeta"


def test_store_empty_output(tmp_path: Path) -> None:
    store = _make_store_task(tmp_path, "btest000a", "")
    chunk = store.read_output_lines("btest000a", None, 999_999, status="completed")
    assert chunk.start_line == 0
    assert chunk.end_line == 0
    assert chunk.has_before is False
    assert chunk.has_after is False
    assert chunk.text == ""
    assert chunk.line_too_large is False


# ---------------------------------------------------------------------------
# _load_sidecar semantics – failed load must not report success
# ---------------------------------------------------------------------------


def test_load_sidecar_returns_false_after_failed_load(tmp_path: Path) -> None:
    """After a corrupted sidecar, _load_sidecar() must return False on re-call,
    not True (the old code set `_sidecar_loaded=True` before checking)."""
    p = tmp_path / "output.log"
    _write(p, "a\nb\n")
    idx_path = p.parent / (p.name + LineIndex.INDEX_SUFFIX)
    idx_path.write_bytes(b"garbage")

    idx = LineIndex(p)
    # First call – corrupt sidecar → should return False.
    result1 = idx._load_sidecar()  # noqa: SLF001
    assert result1 is False

    # Second call – cached, still False.
    result2 = idx._load_sidecar()  # noqa: SLF001
    assert result2 is False


def test_load_sidecar_returns_false_when_no_sidecar(tmp_path: Path) -> None:
    """Missing sidecar → False, and repeated calls stay False."""
    p = tmp_path / "output.log"
    _write(p, "hello\n")

    idx = LineIndex(p)
    assert idx._load_sidecar() is False  # noqa: SLF001
    assert idx._load_sidecar() is False  # noqa: SLF001


def test_load_sidecar_returns_true_after_valid_sidecar(tmp_path: Path) -> None:
    """A valid sidecar round-trips and _load_sidecar reports True."""
    p = tmp_path / "output.log"
    _write(p, "x\ny\n")
    # Build and persist a valid sidecar.
    idx1 = LineIndex(p)
    idx1.refresh()
    assert idx1.line_count == 2

    # New instance loads the sidecar.
    idx2 = LineIndex(p)
    assert idx2._load_sidecar() is True  # noqa: SLF001
    assert idx2._load_sidecar() is True  # noqa: SLF001  (cached)
    assert idx2._scanned_to > 0  # noqa: SLF001


def test_load_sidecar_bad_offsets_return_false(tmp_path: Path) -> None:
    """Sidecar with non-monotonic offsets → False, not True."""
    from array import array as _array

    p = tmp_path / "output.log"
    _write(p, "a\nb\nc\n")
    idx_path = p.parent / (p.name + LineIndex.INDEX_SUFFIX)

    # Craft sidecar with non-monotonic offsets (2, 0 instead of 0, 2).
    bad: _array[int] = _array("Q", [6, 0, 2, 0])
    idx_path.write_bytes(bad.tobytes())

    idx = LineIndex(p)
    assert idx._load_sidecar() is False  # noqa: SLF001
    assert idx._load_sidecar() is False  # noqa: SLF001


# ---------------------------------------------------------------------------
# _save_sidecar fd ownership – no double-close risk
# ---------------------------------------------------------------------------


def test_save_sidecar_with_replace_failure_no_double_close(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If os.replace fails, fd is already closed (by os.fdopen) and temp is cleaned."""
    p = tmp_path / "output.log"
    _write(p, "a\nb\n")
    idx = LineIndex(p)
    idx.refresh()
    assert idx.line_count == 2

    # Remove the valid sidecar so we can test the failure path.
    idx_path = p.parent / (p.name + LineIndex.INDEX_SUFFIX)
    if idx_path.exists():
        idx_path.unlink()

    # Patch os.replace to always fail.
    original_replace = os.replace

    def _fail_replace(src: str, dst: str) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", _fail_replace)

    # _save_sidecar should not raise, and no temp files should remain.
    idx._save_sidecar()  # noqa: SLF001

    monkeypatch.setattr(os, "replace", original_replace)

    # The sidecar should NOT exist (replace failed).
    assert not idx_path.exists()
    # No leftover .lidx.tmp files.
    tmp_files = list(p.parent.glob("*.lidx.tmp"))
    assert tmp_files == [], f"temp files should be cleaned up: {tmp_files}"


def test_save_sidecar_produces_valid_sidecar(tmp_path: Path) -> None:
    """Normal _save_sidecar + reload via a fresh instance works end-to-end."""
    p = tmp_path / "output.log"
    _write(p, "foo\nbar\nbaz\n")
    idx1 = LineIndex(p)
    idx1.refresh()

    idx2 = LineIndex(p)
    idx2.refresh()
    assert idx2.line_count == 3
    assert idx2.line_byte_offset(0) == 0
    assert idx2.line_byte_offset(1) == 4
    assert idx2.line_byte_offset(2) == 8

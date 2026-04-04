from __future__ import annotations

import pytest
from kaos.path import KaosPath

from kimi_cli.tools.file.replace import (
    AppendOp,
    DeleteOp,
    EditParams,
    InsertAfterOp,
    InsertBeforeOp,
    PatchOp,
    PrependOp,
    ReplaceLinesOp,
    ReplaceOp,
)


async def test_append_operation(edit_tool, temp_work_dir: KaosPath):
    file_path = temp_work_dir / "append.txt"
    await file_path.write_text("hello")

    result = await edit_tool(EditParams(path=str(file_path), edit=[AppendOp(content=" world")]))

    assert not result.is_error
    assert await file_path.read_text() == "hello world"


async def test_prepend_operation(edit_tool, temp_work_dir: KaosPath):
    file_path = temp_work_dir / "prepend.txt"
    await file_path.write_text("world")

    result = await edit_tool(EditParams(path=str(file_path), edit=[PrependOp(content="hello ")]))

    assert not result.is_error
    assert await file_path.read_text() == "hello world"


async def test_delete_operation(edit_tool, temp_work_dir: KaosPath):
    file_path = temp_work_dir / "delete.txt"
    await file_path.write_text("alpha beta gamma")

    result = await edit_tool(EditParams(path=str(file_path), edit=[DeleteOp(old=" beta")]))

    assert not result.is_error
    assert await file_path.read_text() == "alpha gamma"


async def test_insert_before_operation(edit_tool, temp_work_dir: KaosPath):
    file_path = temp_work_dir / "insert_before.txt"
    await file_path.write_text("a\nb\nc\n")

    result = await edit_tool(
        EditParams(
            path=str(file_path),
            edit=[InsertBeforeOp(anchor="b\n", content="before-b\n")],
        )
    )

    assert not result.is_error
    assert await file_path.read_text() == "a\nbefore-b\nb\nc\n"


async def test_insert_after_negative_occurrence(edit_tool, temp_work_dir: KaosPath):
    file_path = temp_work_dir / "insert_after.txt"
    await file_path.write_text("tag\nbody\ntag\n")

    result = await edit_tool(
        EditParams(
            path=str(file_path),
            edit=[InsertAfterOp(anchor="tag\n", content="after-last\n", occurrence=-1)],
        )
    )

    assert not result.is_error
    assert await file_path.read_text() == "tag\nbody\ntag\nafter-last\n"


async def test_replace_lines_negative_indices(edit_tool, temp_work_dir: KaosPath):
    file_path = temp_work_dir / "replace_lines.txt"
    await file_path.write_text("1\n2\n3\n4\n")

    result = await edit_tool(
        EditParams(
            path=str(file_path),
            edit=[
                ReplaceLinesOp(
                    start_line=-2,
                    end_line=-1,
                    content="three\nfour\n",
                )
            ],
        )
    )

    assert not result.is_error
    assert await file_path.read_text() == "1\n2\nthree\nfour\n"


async def test_patch_operation(edit_tool, temp_work_dir: KaosPath):
    file_path = temp_work_dir / "patch.txt"
    await file_path.write_text("one\ntwo\nthree\n")

    patch = """@@ -1,3 +1,4 @@
 one
 two
+two-and-half
 three
"""
    result = await edit_tool(EditParams(path=str(file_path), edit=[PatchOp(patch=patch)]))

    assert not result.is_error
    assert await file_path.read_text() == "one\ntwo\ntwo-and-half\nthree\n"


async def test_edit_requires_explicit_kind(temp_work_dir: KaosPath):
    file_path = temp_work_dir / "legacy.txt"
    await file_path.write_text("hello world")

    with pytest.raises(ValueError, match="kind"):
        EditParams.model_validate(
            {
                "path": str(file_path),
                "edit": {"old": "world", "new": "there"},
            }
        )


async def test_replace_fuzzy_hint_on_mismatch(edit_tool, temp_work_dir: KaosPath):
    """When a replace target is not found, the error should include a fuzzy-match hint."""
    file_path = temp_work_dir / "fuzzy.txt"
    await file_path.write_text("def hello_world():\n    print('hello')\n    return True\n")

    # Slightly wrong old string — extra space and different quotes
    result = await edit_tool(
        EditParams(
            path=str(file_path),
            edit=[ReplaceOp(old="def hello_world( ):\n    print('hello')", new="replaced")],
        )
    )

    assert result.is_error
    assert "could not find the target string" in result.message
    # Fuzzy-match hint is included in the error message
    assert "Closest match" in result.message
    assert "similarity" in result.message


async def test_delete_fuzzy_hint_on_mismatch(edit_tool, temp_work_dir: KaosPath):
    """When a delete target is not found, the error should include a fuzzy-match hint."""
    file_path = temp_work_dir / "fuzzy_delete.txt"
    await file_path.write_text("line one\nline two\nline three\n")

    result = await edit_tool(
        EditParams(
            path=str(file_path),
            edit=[DeleteOp(old="line  two")],  # extra space
        )
    )

    assert result.is_error
    assert "could not find the target string" in result.message
    assert "Closest match" in result.message

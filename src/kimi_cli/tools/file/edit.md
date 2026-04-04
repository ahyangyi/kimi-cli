Edit a text file using structured edit operations.

**Tips:**
- Only use this tool on text files.
- You can provide a single edit operation or a list of operations in one call.
- Supported edit kinds are `replace`, `append`, `prepend`, `delete`, `insert_before`, `insert_after`, `replace_lines`, and `patch`.
- Replace operations must use `kind: "replace"`.
- `replace_lines` uses 1-based inclusive line numbers; negative values count backward from the end of the file.
- `patch` accepts unified diff or hunk-only patch text and must apply cleanly to the current file.
- You should prefer this tool over WriteFile tool and Shell `sed` command when you want focused edits instead of rewriting the whole file.

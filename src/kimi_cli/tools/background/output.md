Retrieve output from a running or completed background task.

Use this after `Shell(run_in_background=true)` when you need to inspect progress or explicitly wait for completion.

Guidelines:
- Prefer relying on automatic completion notifications. Use this tool only when you need task output before the automatic notification arrives.
- Use `block=true` to wait for completion or timeout.
- Use `block=false` for a non-blocking status and output check.
- `timeout` (default 30s, max 3600s): How long to wait when `block=true`. If the task does not finish within this time, `retrieval_status` will be `timeout` (task still running). With `block=false`, a still-running task yields `not_ready`.
- `offset`: 0-based **line** offset. By default, the output preview shows the **tail** (last lines fitting within ~32 KiB). Set `offset=0` to read from the beginning instead.
- The preview is returned as whole lines capped at 32 KiB. If a single line exceeds the cap it is omitted and a hint directs you to use `ReadFile` on the `output_path`.
- The response includes `output_preview_start_line`, `output_preview_end_line` (exclusive — the range is `[start, end)`), `output_has_before`, `output_has_after`, and `output_next_offset` for line-level pagination. Use `output_next_offset` as the next `offset` value to continue forward.
- When the preview is truncated, use `ReadFile` with the returned `output_path` to inspect the full log in pages.
- This tool works with the generic background task system and should remain the primary read path for future task types, not just bash.
- When context budget is tight, always pass `offset` (from the previous `output_next_offset`) to read only new lines instead of re-reading the full output.

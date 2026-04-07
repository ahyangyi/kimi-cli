Write input to a running interactive background task's stdin.

Use this to send data to a background task that was started with `interactive=true`. The task must be in a non-terminal state. A common use case is sending NDJSON messages to a Claude Code process running in `--input-format stream-json` mode.

Parameters:
- `task_id` (required): The background task ID to write to.
- `input` (required): The text to write to the task's stdin.
- `append_newline` (default true): Whether to append a newline character after the input. Set to `true` for NDJSON protocols where each message must be on its own line.

Guidelines:
- The task must have been started with `interactive=true` on the Shell call.
- Writing is asynchronous — input is queued and delivered to the task's stdin shortly (within ~200ms).
- After writing, use `TaskOutput(task_id=..., block=false)` to poll for new output. The response includes `output_next_offset` which you can pass as `offset` on the next `TaskOutput` call to read only new lines.
- If the child process has already exited, the write will be silently dropped by the relay.
- This tool does **not** wait for the task to process the input or produce output. Pair it with `TaskOutput` to observe results.

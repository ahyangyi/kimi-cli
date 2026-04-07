from __future__ import annotations

import asyncio
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from kimi_cli.utils.logging import logger
from kimi_cli.utils.subprocess_env import get_clean_env

from .models import TaskControl, TaskRuntime
from .store import BackgroundTaskStore

STDIN_QUEUE_DIR = "stdin_queue"
STDIN_LOG_FILE = "stdin.log"
MAX_STDIN_LOG_BYTES = 10 * 1024 * 1024  # 10 MiB


async def _stdin_relay_loop(
    task_dir: Path,
    process: asyncio.subprocess.Process,
    stop_event: asyncio.Event,
    poll_interval_ms: int = 200,
) -> None:
    """Poll ``stdin_queue/`` for message files and relay them to *process.stdin*.

    Each ``.msg`` file is read, written to the child's stdin pipe, appended to
    ``stdin.log`` for auditing, then deleted.  Files are processed in filename
    order (callers use a nanosecond-timestamp prefix for ordering).
    """
    queue_dir = task_dir / STDIN_QUEUE_DIR
    log_path = task_dir / STDIN_LOG_FILE
    while not stop_event.is_set():
        await asyncio.sleep(poll_interval_ms / 1000)
        if stop_event.is_set() or process.returncode is not None:
            return
        if process.stdin is None:
            return
        try:
            pending = sorted(p for p in queue_dir.iterdir() if p.is_file() and p.suffix == ".msg")
        except FileNotFoundError:
            logger.warning("stdin relay: queue directory removed, stopping")
            return
        except OSError:
            logger.exception("stdin relay: failed to list queue directory")
            continue
        for msg_file in pending:
            if stop_event.is_set() or process.returncode is not None:
                break
            try:
                data = msg_file.read_bytes()
            except OSError:
                logger.warning("stdin relay: failed to read %s, skipping", msg_file.name)
                continue
            if process.returncode is not None:
                msg_file.unlink(missing_ok=True)
                return
            try:
                process.stdin.write(data)
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                logger.warning("stdin relay: pipe write failed, stopping")
                return
            # Audit log (best-effort, capped at MAX_STDIN_LOG_BYTES)
            try:
                if not log_path.exists() or log_path.stat().st_size < MAX_STDIN_LOG_BYTES:
                    with log_path.open("ab") as f:
                        f.write(data)
            except OSError:
                pass
            try:
                msg_file.unlink()
            except OSError:
                logger.warning("stdin relay: failed to unlink %s, may re-send", msg_file.name)


def terminate_process_tree_windows(pid: int, *, force: bool) -> None:
    args = ["taskkill", "/PID", str(pid), "/T"]
    if force:
        args.append("/F")
    subprocess.run(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


def finalize_task_runtime(
    runtime: TaskRuntime,
    *,
    control: TaskControl,
    returncode: int,
    finished_at: float,
    process_exited_at: float | None,
    timed_out: bool,
    timeout_reason: str | None,
) -> TaskRuntime:
    final_runtime = runtime.model_copy()
    final_runtime.finished_at = finished_at
    final_runtime.updated_at = finished_at
    final_runtime.exit_code = returncode
    final_runtime.heartbeat_at = finished_at
    if timed_out:
        final_runtime.status = "failed"
        final_runtime.interrupted = True
        final_runtime.timed_out = True
        final_runtime.failure_reason = timeout_reason
    elif control.kill_requested_at is not None and (
        process_exited_at is None or control.kill_requested_at <= process_exited_at
    ):
        final_runtime.status = "killed"
        final_runtime.interrupted = True
        final_runtime.failure_reason = control.kill_reason or "Killed"
    elif returncode == 0:
        final_runtime.status = "completed"
        final_runtime.failure_reason = None
    else:
        final_runtime.status = "failed"
        final_runtime.failure_reason = f"Command failed with exit code {returncode}"
    return final_runtime


async def run_background_task_worker(
    task_dir: Path,
    *,
    heartbeat_interval_ms: int = 5000,
    control_poll_interval_ms: int = 500,
    kill_grace_period_ms: int = 2000,
) -> None:
    task_dir = task_dir.expanduser().resolve()
    task_id = task_dir.name
    store = BackgroundTaskStore(task_dir.parent)
    spec = store.read_spec(task_id)
    runtime = store.read_runtime(task_id)

    runtime.status = "starting"
    runtime.worker_pid = os.getpid()
    runtime.started_at = time.time()
    runtime.heartbeat_at = runtime.started_at
    runtime.updated_at = runtime.started_at
    store.write_runtime(task_id, runtime)

    control = store.read_control(task_id)
    if control.kill_requested_at is not None:
        runtime.status = "killed"
        runtime.interrupted = True
        runtime.finished_at = time.time()
        runtime.updated_at = runtime.finished_at
        runtime.failure_reason = control.kill_reason or "Killed before command start"
        store.write_runtime(task_id, runtime)
        return

    if spec.command is None or spec.shell_path is None or spec.cwd is None:
        runtime.status = "failed"
        runtime.finished_at = time.time()
        runtime.updated_at = runtime.finished_at
        runtime.failure_reason = "Task spec is incomplete for bash worker"
        store.write_runtime(task_id, runtime)
        return

    if spec.interactive and os.name == "nt":
        runtime.status = "failed"
        runtime.finished_at = time.time()
        runtime.updated_at = runtime.finished_at
        runtime.failure_reason = "Interactive background tasks are not supported on Windows"
        store.write_runtime(task_id, runtime)
        return

    process: asyncio.subprocess.Process | None = None
    control_task: asyncio.Task[None] | None = None
    heartbeat_task: asyncio.Task[None] | None = None
    stdin_relay_task: asyncio.Task[None] | None = None
    stop_event = asyncio.Event()

    # Register SIGTERM handler to initiate graceful shutdown
    def _sigterm_handler(signum: int, frame: object) -> None:
        stop_event.set()

    original_sigterm = signal.getsignal(signal.SIGTERM)
    if os.name != "nt":
        signal.signal(signal.SIGTERM, _sigterm_handler)

    kill_sent_at: float | None = None
    timed_out = False
    timeout_reason: str | None = None
    process_exited_at: float | None = None

    async def _heartbeat_loop() -> None:
        while not stop_event.is_set():
            await asyncio.sleep(heartbeat_interval_ms / 1000)
            if stop_event.is_set():
                return
            try:
                current = await asyncio.to_thread(store.read_runtime, task_id)
                if current.finished_at is not None:
                    return
                current.heartbeat_at = time.time()
                current.updated_at = current.heartbeat_at
                await asyncio.to_thread(store.write_runtime, task_id, current)
            except Exception:
                logger.exception("Heartbeat write failed")

    async def _terminate_process(force: bool = False) -> None:
        nonlocal kill_sent_at
        if process is None or process.returncode is not None:
            return

        # Graceful EOF shutdown for interactive tasks: close stdin first so
        # the child process sees EOF and can exit cleanly.
        if spec.interactive and not force and process.stdin is not None:
            try:
                process.stdin.close()
                await asyncio.wait_for(process.wait(), timeout=1.0)
                return  # exited cleanly after EOF
            except TimeoutError:
                pass  # proceed to SIGTERM
            except Exception:
                pass

        kill_sent_at = kill_sent_at or time.time()

        try:
            if os.name == "nt":
                terminate_process_tree_windows(process.pid, force=force)
                return

            target_pgid = process.pid
            if force:
                os.killpg(target_pgid, signal.SIGKILL)
            else:
                os.killpg(target_pgid, signal.SIGTERM)
        except OSError:
            pass

    async def _control_loop() -> None:
        nonlocal kill_sent_at
        while not stop_event.is_set():
            await asyncio.sleep(control_poll_interval_ms / 1000)
            if stop_event.is_set():
                return
            current_control = await asyncio.to_thread(store.read_control, task_id)
            if current_control.kill_requested_at is not None:
                await _terminate_process(force=current_control.force)
                if (
                    kill_sent_at is not None
                    and process is not None
                    and process.returncode is None
                    and time.time() - kill_sent_at >= kill_grace_period_ms / 1000
                ):
                    await _terminate_process(force=True)

    try:
        output_path = store.output_path(task_id)
        with output_path.open("ab") as output_file:
            spawn_kwargs: dict[str, Any] = {
                "stdin": asyncio.subprocess.PIPE if spec.interactive else subprocess.DEVNULL,
                "stdout": output_file,
                "stderr": output_file,
                "cwd": spec.cwd,
                "env": get_clean_env(),
            }
            if os.name == "nt":
                spawn_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            else:
                spawn_kwargs["start_new_session"] = True

            args = (
                (spec.shell_path, "-command", spec.command)
                if spec.shell_name == "Windows PowerShell"
                else (spec.shell_path, "-c", spec.command)
            )
            process = await asyncio.create_subprocess_exec(*args, **spawn_kwargs)

            runtime = store.read_runtime(task_id)
            runtime.status = "running"
            runtime.child_pid = process.pid
            runtime.child_pgid = process.pid if os.name != "nt" else None
            if spec.interactive:
                stdin_queue_dir = task_dir / STDIN_QUEUE_DIR
                stdin_queue_dir.mkdir(exist_ok=True)
                runtime.stdin_ready = True
            runtime.updated_at = time.time()
            runtime.heartbeat_at = runtime.updated_at
            store.write_runtime(task_id, runtime)
            last_known_runtime = runtime

            heartbeat_task = asyncio.create_task(_heartbeat_loop())
            control_task = asyncio.create_task(_control_loop())
            if spec.interactive:
                stdin_relay_task = asyncio.create_task(
                    _stdin_relay_loop(task_dir, process, stop_event)
                )
            if spec.timeout_s is None:
                returncode = await process.wait()
            else:
                try:
                    returncode = await asyncio.wait_for(process.wait(), timeout=spec.timeout_s)
                except TimeoutError:
                    timed_out = True
                    timeout_reason = f"Command timed out after {spec.timeout_s}s"
                    await _terminate_process(force=False)
                    try:
                        returncode = await asyncio.wait_for(
                            process.wait(),
                            timeout=kill_grace_period_ms / 1000,
                        )
                    except TimeoutError:
                        await _terminate_process(force=True)
                        returncode = await process.wait()
            process_exited_at = time.time()
    except Exception as exc:
        logger.exception("Background task worker failed")
        runtime = store.read_runtime(task_id)
        runtime.status = "failed"
        runtime.finished_at = time.time()
        runtime.updated_at = runtime.finished_at
        runtime.failure_reason = str(exc)
        store.write_runtime(task_id, runtime)
        return
    finally:
        stop_event.set()
        if os.name != "nt":
            signal.signal(signal.SIGTERM, original_sigterm)
        for task in (heartbeat_task, control_task, stdin_relay_task):
            if task is not None:
                task.cancel()
                try:
                    await task
                except Exception:
                    logger.exception("Background helper task failed")
                except asyncio.CancelledError:
                    pass

    control = store.read_control(task_id)
    runtime = finalize_task_runtime(
        last_known_runtime,
        control=control,
        returncode=returncode,
        finished_at=time.time(),
        process_exited_at=process_exited_at,
        timed_out=timed_out,
        timeout_reason=timeout_reason,
    )
    store.write_runtime(task_id, runtime)

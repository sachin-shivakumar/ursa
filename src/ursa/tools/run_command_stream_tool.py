# ..tools/run_command_stream_tool.py
import queue
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, List, TypedDict

from langchain.tools import ToolRuntime
from langchain_core.tools import tool
from rich import get_console

from ursa.agents.base import AgentContext
from ursa.prompt_library.execution_prompts import get_safety_prompt
from ursa.util.types import AsciiStr

console = get_console()


class SafetyAssessment(TypedDict):
    is_safe: bool
    reason: str


# job_id -> {"proc": Popen, "q": Queue[str], "cmd": str}
_JOBS: Dict[str, Dict[str, Any]] = {}


def _reader_thread(stream, q: "queue.Queue[str]", prefix: str) -> None:
    try:
        for line in iter(stream.readline, ""):
            if not line:
                break
            q.put(prefix + line)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _safety_check(query: str, runtime: ToolRuntime[AgentContext]) -> SafetyAssessment:
    workspace_dir = Path(runtime.context.workspace)

    if runtime.store is not None:
        edited_files = [item.key for item in runtime.store.search(("workspace", "file_edit"), limit=1000)]
        safe_codes = [item.key for item in runtime.store.search(("workspace", "safe_codes"), limit=1000)]
    else:
        edited_files, safe_codes = [], []

    llm = runtime.context.llm
    return llm.with_structured_output(SafetyAssessment).invoke(
        get_safety_prompt(query, safe_codes, edited_files)
    )


@tool
def run_command_stream_start(query: AsciiStr, runtime: ToolRuntime[AgentContext]) -> Dict[str, Any]:
    """Start a shell command in the workspace and stream output via run_command_stream_poll."""
    safety_result = _safety_check(query, runtime)
    if not safety_result["is_safe"]:
        msg = f"[UNSAFE] `{query}`\nReason: {safety_result['reason']}"
        console.print("[bold red][WARNING][/bold red] Command deemed unsafe:", query)
        console.print("[bold red][WARNING][/bold red] REASON:", msg)
        return {"ok": False, "error": msg}

    console.print(f"[green]Command passed safety check:[/green] {query}\nFor reason: {safety_result['reason']}")
    workspace_dir = Path(runtime.context.workspace)

    proc = subprocess.Popen(
        query,
        text=True,
        shell=True,
        cwd=workspace_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
        universal_newlines=True,
    )

    job_id = str(uuid.uuid4())
    q: "queue.Queue[str]" = queue.Queue()

    threading.Thread(target=_reader_thread, args=(proc.stdout, q, ""), daemon=True).start()
    threading.Thread(target=_reader_thread, args=(proc.stderr, q, "STDERR: "), daemon=True).start()

    _JOBS[job_id] = {"proc": proc, "q": q, "cmd": query}
    return {"ok": True, "job_id": job_id}


@tool
def run_command_stream_poll(job_id: AsciiStr, runtime: ToolRuntime[AgentContext]) -> Dict[str, Any]:
    """Poll for new output lines from a running job."""
    job = _JOBS.get(job_id)
    if not job:
        return {"ok": False, "error": f"Unknown job_id: {job_id}", "lines": [], "done": True, "returncode": None}

    proc: subprocess.Popen = job["proc"]
    q: "queue.Queue[str]" = job["q"]

    lines: List[str] = []
    for _ in range(200):
        try:
            lines.append(q.get_nowait())
        except queue.Empty:
            break

    done = proc.poll() is not None
    return {"ok": True, "lines": lines, "done": done, "returncode": proc.returncode}


@tool
def run_command_stream_cancel(job_id: AsciiStr, runtime: ToolRuntime[AgentContext]) -> Dict[str, Any]:
    """Cancel a running job."""
    job = _JOBS.get(job_id)
    if not job:
        return {"ok": False, "error": f"Unknown job_id: {job_id}"}

    proc: subprocess.Popen = job["proc"]
    try:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception as e:
        return {"ok": False, "error": str(e)}
    finally:
        _JOBS.pop(job_id, None)

    return {"ok": True}

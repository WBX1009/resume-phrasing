import subprocess
import sys
from pathlib import Path
from uuid import uuid4

from backend.common.config import settings

SANDBOX_DIR = Path(settings.AGENT_SANDBOX_DIR).resolve()
RUN_DIR = SANDBOX_DIR / ".python_runs"
RUN_DIR.mkdir(parents=True, exist_ok=True)


def execute_python_sandbox(code: str, timeout_seconds: int = 10) -> str:
    """Execute Python code inside the configured local agent sandbox.

    This tool is intended for local development and personal-assistant workflows
    such as calculations, small data analysis, and model prototyping. It is a
    high-risk code execution tool and should be gated by HITL unless explicitly
    disabled for development testing.
    """

    if not code.strip():
        return "No Python code was provided."

    timeout = max(1, min(int(timeout_seconds), 30))
    script_path = RUN_DIR / f"tool_run_{uuid4().hex}.py"
    script_path.write_text(code, encoding="utf-8")

    try:
        completed = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(SANDBOX_DIR),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return (
            f"Python execution timed out after {timeout} seconds.\n"
            f"stdout:\n{exc.stdout or ''}\n"
            f"stderr:\n{exc.stderr or ''}"
        )
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass

    stdout = completed.stdout.strip()
    stderr = completed.stderr.strip()
    parts = [f"exit_code: {completed.returncode}"]
    if stdout:
        parts.append(f"stdout:\n{stdout}")
    if stderr:
        parts.append(f"stderr:\n{stderr}")
    if not stdout and not stderr:
        parts.append("stdout: <empty>")
    return "\n".join(parts)

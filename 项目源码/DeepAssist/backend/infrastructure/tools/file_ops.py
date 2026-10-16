from pathlib import Path

from backend.common.config import settings
from backend.common.logger import get_logger

logger = get_logger(__name__)

SANDBOX_DIR = Path(settings.AGENT_SANDBOX_DIR).resolve()
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_sandbox_path(relative_path: str) -> Path:
    target_path = (SANDBOX_DIR / (relative_path or ".")).resolve()
    try:
        target_path.relative_to(SANDBOX_DIR)
    except ValueError as exc:
        raise PermissionError("Path escapes the agent sandbox") from exc
    return target_path


def read_local_file(file_name: str) -> str:
    """Read a UTF-8 text file inside the configured agent sandbox."""

    try:
        target_path = _resolve_sandbox_path(file_name)

        if not target_path.exists():
            return f"File does not exist: {file_name}"
        if not target_path.is_file():
            return f"Path is not a file: {file_name}"

        return target_path.read_text(encoding="utf-8")
    except PermissionError:
        return "Security blocked: path escapes the agent sandbox."
    except Exception as exc:
        return f"Failed to read file: {exc}"


def write_local_file(file_name: str, content: str) -> str:
    """Write UTF-8 text to a file inside the configured agent sandbox."""

    try:
        target_path = _resolve_sandbox_path(file_name)
        target_path.parent.mkdir(parents=True, exist_ok=True)

        target_path.write_text(content, encoding="utf-8")
        logger.info("[Tool] Agent wrote sandbox file: %s", file_name)
        return f"Successfully wrote file: {file_name}"
    except PermissionError:
        return "Security blocked: path escapes the agent sandbox."
    except Exception as exc:
        return f"Failed to write file: {exc}"


def list_sandbox_files(path: str = ".") -> str:
    """List files and directories under the configured agent sandbox."""

    try:
        target_path = _resolve_sandbox_path(path)

        if not target_path.exists() or not target_path.is_dir():
            return f"Directory does not exist or is not a directory: {path}"

        items = []
        for item in sorted(target_path.iterdir(), key=lambda value: value.name):
            prefix = "[DIR] " if item.is_dir() else "[FILE]"
            size = f" ({item.stat().st_size} bytes)" if item.is_file() else ""
            items.append(f"{prefix} {item.name}{size}")

        if not items:
            return f"Directory is empty: {path}"

        return "Sandbox directory contents:\n" + "\n".join(items)
    except PermissionError:
        return "Security blocked: path escapes the agent sandbox."
    except Exception as exc:
        return f"Failed to list directory: {exc}"

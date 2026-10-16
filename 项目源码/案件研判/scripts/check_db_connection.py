from __future__ import annotations

from pathlib import Path
import sys

from dotenv import dotenv_values


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import load_config
from src.data_loader import _connect


def _mask(value: str | None) -> str:
    if not value:
        return "missing"
    return f"present(len={len(value)})"


def main() -> int:
    env_path = ROOT_DIR / ".env"
    raw = env_path.read_bytes() if env_path.exists() else b""
    values = dotenv_values(env_path, encoding="utf-8-sig") if env_path.exists() else {}
    has_bom = raw.startswith(b"\xef\xbb\xbf")
    config = load_config()

    print(f"python_executable={sys.executable}")
    print(f"python_version={sys.version.split()[0]}")
    print(f"env_file_exists={env_path.exists()}")
    print(f"env_utf8_bom={has_bom}")
    print("env_keys=" + ",".join(sorted(key.lstrip("\ufeff") for key in values if key)))
    print(f"db_complete={config.db.is_complete}")
    print(f"db_host={_mask(config.db.host)}")
    print(f"db_port={config.db.port if config.db.port else 'missing'}")
    print(f"db_user={_mask(config.db.user)}")
    print(f"db_password={_mask(config.db.password)}")
    print(f"db_database={_mask(config.db.database)}")
    print(f"db_schema={config.db.schema}")
    print(f"db_ssl_mode={config.db.ssl_mode}")

    try:
        with _connect(config) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
        print(f"db_connect=success result={row}")
        return 0
    except Exception as exc:
        print(f"db_connect=failure type={type(exc).__name__}")
        print(f"db_error={exc}")
        diag = getattr(exc, "diag", None)
        if diag is not None:
            for attr in ("severity", "message_primary", "message_detail", "message_hint", "sqlstate"):
                value = getattr(diag, attr, None)
                if value:
                    print(f"db_diag_{attr}={value}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

"""Run the same case graph used by the HTTP service."""
import argparse
import json
from pathlib import Path
import time

from .case_api import build_service
from .case_run_service import TERMINAL


def main(argv=None):
    parser = argparse.ArgumentParser(description="LangGraph evidence-first case investigation")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--offline", action="store_true", help="explicit rule-only mode; semantic gaps remain unresolved")
    parser.add_argument("--run-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    service = build_service(repository_root=args.repository, storage_root=args.storage, offline=args.offline)
    try:
        if args.resume:
            if not args.run_id:
                parser.error("--resume requires --run-id")
            if args.run_id not in service._futures:
                record = service.status(args.run_id)
                if not record or record["status"] not in TERMINAL:
                    service.resume(args.run_id)
            run_id = args.run_id
        else:
            run_id = service.submit(args.case_id, run_id=args.run_id)["run_id"]
        while service.status(run_id)["status"] not in TERMINAL:
            time.sleep(.05)
        record = service.status(run_id)
        result = service.report(run_id) or record
        encoded = json.dumps(result, ensure_ascii=False, indent=2)
        if args.report:
            if record["status"] in {"success", "partial"}:
                from .case_report import write_report
                write_report(result, args.report)
        print(encoded)
        return 1 if record["status"] in {"failed", "cancelled"} else 0
    finally:
        service.close()


if __name__ == "__main__":
    raise SystemExit(main())

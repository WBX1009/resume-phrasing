from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

try:
    from .dataset_catalog import DATASET_SPECS, DatasetSpec, selected_specs
except ImportError:  # pragma: no cover - direct script execution
    from dataset_catalog import DATASET_SPECS, DatasetSpec, selected_specs


DEFAULT_DATASETS = [
    "beir_nfcorpus",
    "beir_fiqa",
    "squad",
    "hotpotqa",
    "gsm8k",
    "bfcl",
    "cmrc2018",
]


def download_dataset(
    spec: DatasetSpec,
    cache_dir: Path,
    sample_size: int,
    force: bool = False,
) -> Dict[str, Any]:
    target_dir = cache_dir / spec.name
    metadata_path = target_dir / "download_metadata.json"
    if target_dir.exists() and metadata_path.exists() and not force:
        return _load_json(metadata_path)

    try:
        from datasets import load_dataset
    except Exception as exc:
        raise RuntimeError(
            "The 'datasets' package is required. Install it with `pip install datasets`."
        ) from exc

    load_kwargs: Dict[str, Any] = {}
    if spec.config:
        load_kwargs["name"] = spec.config
    if spec.split:
        load_kwargs["split"] = spec.split

    dataset = load_dataset(spec.hf_id, **load_kwargs)
    original_count = len(dataset) if hasattr(dataset, "__len__") else None
    if sample_size > 0 and original_count and original_count > sample_size:
        dataset = dataset.select(range(sample_size))

    target_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(target_dir / "dataset"))

    metadata = {
        "name": spec.name,
        "hf_id": spec.hf_id,
        "config": spec.config,
        "split": spec.split,
        "suite": spec.suite,
        "source_url": spec.source_url,
        "license": spec.license,
        "sample_size": sample_size,
        "saved_count": len(dataset) if hasattr(dataset, "__len__") else None,
        "original_count": original_count,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "notes": spec.notes,
        "local_path": str((target_dir / "dataset").as_posix()),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metadata


def write_manifest(items: Iterable[Dict[str, Any]], manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DeepAssist public benchmark dataset manifest",
        f"generated_at: {datetime.now(timezone.utc).isoformat()}",
        "datasets:",
    ]
    for item in items:
        lines.extend(
            [
                f"  - name: {item['name']}",
                f"    suite: {item['suite']}",
                f"    hf_id: {item['hf_id']}",
                f"    config: {item.get('config') or ''}",
                f"    split: {item.get('split') or ''}",
                f"    sample_size: {item.get('sample_size')}",
                f"    saved_count: {item.get('saved_count')}",
                f"    source_url: {item.get('source_url')}",
                f"    license: {item.get('license')}",
                f"    local_path: {item.get('local_path')}",
            ]
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download public benchmark datasets for DeepAssist evaluation."
    )
    parser.add_argument(
        "--cache-dir",
        default="data/eval_datasets/raw",
        help="Directory for HuggingFace save_to_disk outputs.",
    )
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help=f"Comma-separated dataset names. Available: {', '.join(sorted(DATASET_SPECS))}",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=300,
        help="Maximum rows saved per dataset; use 0 for full split.",
    )
    parser.add_argument(
        "--manifest-path",
        default="evals/dataset_manifest.yaml",
        help="Manifest file written after successful downloads.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing datasets.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cache_dir = Path(args.cache_dir)
    names = [item.strip() for item in args.datasets.split(",") if item.strip()]
    specs = selected_specs(names)

    results: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    for spec in specs:
        try:
            results.append(
                download_dataset(
                    spec=spec,
                    cache_dir=cache_dir,
                    sample_size=max(int(args.sample_size), 0),
                    force=bool(args.force),
                )
            )
        except Exception as exc:
            failures.append(
                {
                    "name": spec.name,
                    "hf_id": spec.hf_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    if results:
        write_manifest(results, Path(args.manifest_path))

    payload = {
        "status": "success" if not failures else "partial_failure",
        "downloaded": results,
        "failures": failures,
        "manifest_path": args.manifest_path if results else None,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

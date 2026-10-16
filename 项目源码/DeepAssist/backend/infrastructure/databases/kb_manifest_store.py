import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backend.domain.interfaces.kb_manifest_store import BaseKnowledgeBaseManifestStore

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

class KnowledgeBaseManifestStore(BaseKnowledgeBaseManifestStore):
    """Persist lightweight collection/file inventory for fast UI and API reads."""

    def __init__(self, manifest_path: str):
        self.path = Path(manifest_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return self._empty_manifest()
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                payload.setdefault("schema_version", 1)
                payload.setdefault("updated_at", _utc_now_iso())
                payload.setdefault("collections", {})
                return payload
        except Exception:
            pass
        return self._empty_manifest()

    def save(self, manifest: Dict[str, Any]) -> None:
        manifest["updated_at"] = _utc_now_iso()
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2)

    def list_collections(self) -> List[Dict[str, Any]]:
        manifest = self.load()
        collections = manifest.get("collections", {})
        if not isinstance(collections, dict):
            return []

        items: List[Dict[str, Any]] = []
        for collection_name, payload in collections.items():
            if not isinstance(payload, dict):
                continue
            files = payload.get("files", {})
            if not isinstance(files, dict):
                files = {}
            items.append(
                {
                    "collection_name": collection_name,
                    "file_count": int(payload.get("file_count", len(files)) or 0),
                    "chunk_count": int(payload.get("chunk_count", 0) or 0),
                    "stores": sorted(set(payload.get("stores", ["vector", "keyword"]))),
                    "consistent": bool(payload.get("consistent", True)),
                    "metadata": payload.get("metadata", {}),
                    "updated_at": payload.get("updated_at") or manifest.get("updated_at"),
                }
            )
        items.sort(key=lambda item: item["collection_name"].lower())
        return items

    def list_files(self, collection_name: str) -> List[Dict[str, Any]]:
        collection = self._collection_payload(collection_name)
        if not collection:
            return []
        files = collection.get("files", {})
        if not isinstance(files, dict):
            return []

        items: List[Dict[str, Any]] = []
        for source_file, payload in files.items():
            if not isinstance(payload, dict):
                continue
            items.append(
                {
                    "source_file": source_file,
                    "chunk_count": int(payload.get("chunk_count", 0) or 0),
                    "vector_chunk_count": int(payload.get("vector_chunk_count", payload.get("chunk_count", 0)) or 0),
                    "keyword_chunk_count": int(payload.get("keyword_chunk_count", payload.get("chunk_count", 0)) or 0),
                    "stores": sorted(set(payload.get("stores", ["vector", "keyword"]))),
                    "consistent": bool(payload.get("consistent", True)),
                    "metadata": payload.get("metadata", {}),
                    "updated_at": payload.get("updated_at") or collection.get("updated_at"),
                }
            )
        items.sort(key=lambda item: item["source_file"].lower())
        return items

    def upsert_file(
        self,
        collection_name: str,
        source_file: str,
        chunk_count: int,
        metadata: Optional[Dict[str, Any]] = None,
        stores: Optional[List[str]] = None,
    ) -> None:
        manifest = self.load()
        collection = self._ensure_collection(manifest, collection_name)
        files = collection.setdefault("files", {})
        if not isinstance(files, dict):
            files = {}
            collection["files"] = files

        stores_value = sorted(set(stores or ["vector", "keyword"]))
        files[source_file] = {
            "source_file": source_file,
            "chunk_count": int(chunk_count),
            "vector_chunk_count": int(chunk_count),
            "keyword_chunk_count": int(chunk_count),
            "stores": stores_value,
            "consistent": True,
            "metadata": metadata or {},
            "updated_at": _utc_now_iso(),
        }
        self._recalculate_collection(collection)
        self.save(manifest)

    def remove_file(self, collection_name: str, source_file: str) -> None:
        manifest = self.load()
        collection = self._ensure_collection(manifest, collection_name)
        files = collection.setdefault("files", {})
        if isinstance(files, dict):
            files.pop(source_file, None)
        self._recalculate_collection(collection)
        self.save(manifest)

    def replace_collection(
        self,
        collection_name: str,
        files: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        stores: Optional[List[str]] = None,
    ) -> None:
        manifest = self.load()
        collection = self._ensure_collection(manifest, collection_name)
        file_map: Dict[str, Dict[str, Any]] = {}
        for item in files:
            source_file = str(item.get("source_file") or "").strip()
            if not source_file:
                continue
            chunk_count = int(item.get("chunk_count", 0) or 0)
            vector_count = int(item.get("vector_chunk_count", chunk_count) or chunk_count)
            keyword_count = int(item.get("keyword_chunk_count", chunk_count) or chunk_count)
            file_map[source_file] = {
                "source_file": source_file,
                "chunk_count": chunk_count,
                "vector_chunk_count": vector_count,
                "keyword_chunk_count": keyword_count,
                "stores": sorted(set(item.get("stores", stores or ["vector", "keyword"]))),
                "consistent": bool(item.get("consistent", vector_count == keyword_count)),
                "metadata": item.get("metadata", {}),
                "updated_at": _utc_now_iso(),
            }
        collection["files"] = file_map
        if metadata is not None:
            collection["metadata"] = metadata
        if stores is not None:
            collection["stores"] = sorted(set(stores))
        self._recalculate_collection(collection)
        self.save(manifest)

    def _empty_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": 1,
            "updated_at": _utc_now_iso(),
            "collections": {},
        }

    def _collection_payload(self, collection_name: str) -> Dict[str, Any]:
        manifest = self.load()
        collections = manifest.get("collections", {})
        if not isinstance(collections, dict):
            return {}
        collection = collections.get(collection_name, {})
        return collection if isinstance(collection, dict) else {}

    def _ensure_collection(
        self,
        manifest: Dict[str, Any],
        collection_name: str,
    ) -> Dict[str, Any]:
        collections = manifest.setdefault("collections", {})
        if not isinstance(collections, dict):
            collections = {}
            manifest["collections"] = collections
        collection = collections.setdefault(
            collection_name,
            {
                "collection_name": collection_name,
                "files": {},
                "file_count": 0,
                "chunk_count": 0,
                "stores": ["keyword", "vector"],
                "consistent": True,
                "metadata": {},
                "updated_at": _utc_now_iso(),
            },
        )
        if not isinstance(collection, dict):
            collection = {}
            collections[collection_name] = collection
        collection.setdefault("collection_name", collection_name)
        collection.setdefault("files", {})
        collection.setdefault("metadata", {})
        return collection

    def _recalculate_collection(self, collection: Dict[str, Any]) -> None:
        files = collection.get("files", {})
        if not isinstance(files, dict):
            files = {}
            collection["files"] = files

        chunk_count = 0
        stores = set(collection.get("stores", []))
        consistent = True
        for payload in files.values():
            if not isinstance(payload, dict):
                continue
            chunk_count += int(payload.get("chunk_count", 0) or 0)
            stores.update(str(store) for store in payload.get("stores", []) if store)
            consistent = consistent and bool(payload.get("consistent", True))

        collection["file_count"] = len(files)
        collection["chunk_count"] = chunk_count
        collection["stores"] = sorted(stores or {"keyword", "vector"})
        collection["consistent"] = consistent
        collection["updated_at"] = _utc_now_iso()

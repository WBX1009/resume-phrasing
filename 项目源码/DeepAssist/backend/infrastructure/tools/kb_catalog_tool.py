from backend.domain.interfaces.kb_catalog import BaseKnowledgeBaseCatalog


class KnowledgeBaseCatalogTool:
    """Read-only toolset for inspecting the connected offline knowledge bases."""

    def __init__(self, kb_catalog: BaseKnowledgeBaseCatalog):
        self.kb_catalog = kb_catalog

    def list_knowledge_base_collections(self) -> str:
        """List the connected knowledge-base collections and their chunk coverage."""
        payload = self.kb_catalog.list_collections()
        collections = payload.get("data", [])
        if not collections:
            return "No knowledge-base collections are currently connected."

        lines = [
            "Connected knowledge-base collections:",
        ]
        for item in collections:
            if not isinstance(item, dict):
                continue
            name = item.get("collection_name", "unknown")
            file_count = int(item.get("file_count", 0))
            chunk_count = int(item.get("chunk_count", 0))
            stores = ", ".join(item.get("stores", [])) or "unavailable"
            lines.append(
                f"- {name}: {file_count} file(s), {chunk_count} chunk(s), stores={stores}"
            )

        lines.append(
            "RAG and Agent retrieval search across all connected collections by default."
        )
        return "\n".join(lines)

    def list_knowledge_base_files(self, collection_name: str = "tech_docs_kb") -> str:
        """List indexed source files inside one knowledge-base collection."""
        payload = self.kb_catalog.list_files(collection_name=collection_name)
        files = payload.get("data", [])
        if not files:
            return f"No indexed files were found in collection '{collection_name}'."

        lines = [f"Indexed files in collection '{collection_name}':"]
        for item in files:
            if not isinstance(item, dict):
                continue
            source_file = item.get("source_file", "unknown")
            chunk_count = int(item.get("chunk_count", 0))
            consistent = bool(item.get("consistent"))
            lines.append(
                f"- {source_file}: {chunk_count} chunk(s), consistent={consistent}"
            )
        return "\n".join(lines)

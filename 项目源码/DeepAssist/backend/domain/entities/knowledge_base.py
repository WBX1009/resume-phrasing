from typing import Any, Dict

from pydantic import BaseModel, Field


class KnowledgeBaseFile(BaseModel):
    """File-level summary for a knowledge-base collection."""

    source_file: str = Field(..., description="Original uploaded file name")
    chunk_count: int = Field(default=0, ge=0, description="Number of chunks in this store")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()

    def update_metadata(self, key: str, value: Any) -> None:
        """Update or add a metadata entry."""
        self.metadata[key] = value

    def increment_chunk_count(self, increment: int = 1) -> None:
        """Increment the stored chunk count."""
        self.chunk_count += increment


class KnowledgeBaseCollectionHealth(BaseModel):
    """Collection-level vector index health summary."""

    collection_name: str = Field(..., description="Knowledge-base collection name")
    expected_segment_id: str | None = Field(
        default=None,
        description="Segment directory id recorded in Chroma manifest",
    )
    actual_segment_dir_present: bool = Field(
        default=False,
        description="Whether the expected Chroma segment directory exists on disk",
    )
    count: int | None = Field(
        default=None,
        ge=0,
        description="Document/chunk count reported by Chroma",
    )
    get_ok: bool = Field(default=False, description="Whether collection.get() succeeded")
    query_ok: bool = Field(default=False, description="Whether collection.query() succeeded")
    whoosh_docs: int | None = Field(
        default=None,
        ge=0,
        description="Stored document/chunk count exported from Whoosh",
    )
    healthy: bool = Field(default=False, description="Overall collection health verdict")
    errors: list[str] = Field(default_factory=list, description="Diagnostic failures or mismatches")
    repaired: bool = Field(default=False, description="Whether repair was attempted in this run")


class KnowledgeBaseHealthReport(BaseModel):
    """Repository-wide knowledge-base health report."""

    vector_db_path: str = Field(..., description="Chroma persistence root")
    keyword_db_path: str = Field(..., description="Whoosh persistence root")
    checked_at: str = Field(..., description="ISO timestamp of the health check")
    collections: list[KnowledgeBaseCollectionHealth] = Field(default_factory=list)
    orphan_segment_dirs: list[str] = Field(default_factory=list)
    quarantined_dirs: list[str] = Field(default_factory=list)

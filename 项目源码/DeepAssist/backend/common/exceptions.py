class DeepAssistError(Exception):
    """Base exception for DeepAssist application errors."""


class ConfigurationError(DeepAssistError):
    """Raised when runtime configuration is incomplete or invalid."""


class DependencyUnavailableError(DeepAssistError):
    """Raised when an optional infrastructure dependency is unavailable."""


class IngestionError(DeepAssistError):
    """Raised when knowledge-base ingestion fails."""

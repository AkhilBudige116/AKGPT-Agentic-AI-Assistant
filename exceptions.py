"""Application-specific exceptions with safe, user-presentable messages."""
from __future__ import annotations


class AKGPTError(Exception):
    """Base error. ``message`` is always safe to show to the client."""

    status_code: int = 500
    code: str = "internal_error"

    def __init__(self, message: str | None = None, *, headers: dict[str, str] | None = None) -> None:
        message = message or self.default_message()
        super().__init__(message)
        self.message = message
        self.headers = headers or {}

    @classmethod
    def default_message(cls) -> str:
        return "An unexpected error occurred."


class ConfigurationError(AKGPTError):
    status_code = 500
    code = "configuration_error"


class ValidationFailed(AKGPTError):
    status_code = 422
    code = "validation_error"


class NotFoundError(AKGPTError):
    status_code = 404
    code = "not_found"

    @classmethod
    def default_message(cls) -> str:
        return "Not found."


class ConflictError(AKGPTError):
    status_code = 409
    code = "conflict"


class PayloadTooLargeError(AKGPTError):
    status_code = 413
    code = "payload_too_large"


class UnsupportedFileError(AKGPTError):
    status_code = 415
    code = "unsupported_file"


class InvalidDocumentError(AKGPTError):
    status_code = 422
    code = "invalid_document"


class RateLimitedError(AKGPTError):
    status_code = 429
    code = "rate_limited"

    @classmethod
    def default_message(cls) -> str:
        return "Too many requests. Please slow down and retry shortly."


class ForbiddenError(AKGPTError):
    status_code = 403
    code = "forbidden"


class ServiceNotConfiguredError(AKGPTError):
    """A required external service (e.g. Groq) has no API key."""

    status_code = 503
    code = "service_not_configured"


class UpstreamError(AKGPTError):
    """An upstream provider (LLM, search, weather) failed."""

    status_code = 502
    code = "upstream_error"


class IndexIncompatibleError(AKGPTError):
    """The persisted FAISS index was built with a different embedding model."""

    status_code = 409
    code = "index_incompatible"


class ToolError(AKGPTError):
    """Raised inside tools; converted to a structured tool result for the model."""

    status_code = 502
    code = "tool_error"

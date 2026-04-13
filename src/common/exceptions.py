from __future__ import annotations


class CommonError(Exception):
    """Base exception for common module."""


class IOErrorBase(CommonError):
    """Base exception for I/O layer."""


class FileReadError(IOErrorBase):
    """Raised when a file cannot be read."""


class CSVSeparatorDetectionError(IOErrorBase):
    """Raised when CSV separator cannot be detected."""


class MissingRequiredFileError(IOErrorBase):
    """Raised when a required file is missing."""


class RepositoryError(IOErrorBase):
    """Raised when repository cannot assemble input bundle."""


class LLMClientError(CommonError):
    """Base exception for LLM client."""


class LLMConfigurationError(LLMClientError):
    """Raised when LLM client configuration is invalid."""


class LLMInvocationError(LLMClientError):
    """Raised when LLM invocation fails."""


class LLMEmptyResponseError(LLMClientError):
    """Raised when LLM returns empty response."""


class FeatureSelectionError(Exception):
    """Base project exception."""


# =========================
# Config / environment
# =========================

class ConfigError(FeatureSelectionError):
    """Configuration or environment error."""


class MissingEnvVariableError(ConfigError):
    """Required environment variable is missing."""


class ProviderConfigurationError(ConfigError):
    """LLM provider is configured incorrectly."""


# =========================
# File / data loading
# =========================

class DataError(FeatureSelectionError):
    """Base data-related error."""


class DataLoadingError(DataError):
    """Failed to load input data."""


class FileReadError(DataLoadingError):
    """Failed to read file from disk."""


class FileWriteError(DataError):
    """Failed to write file to disk."""


class MissingRequiredFileError(DataLoadingError):
    """Required file is missing."""


class CSVDelimiterDetectionError(DataLoadingError):
    """Could not detect CSV delimiter."""


class DatasetValidationError(DataError):
    """Dataset has invalid structure."""


class TableNotFoundError(DataError):
    """Requested table does not exist."""


class ColumnNotFoundError(DataError):
    """Requested column does not exist."""


# =========================
# Schema / joins
# =========================

class SchemaError(FeatureSelectionError):
    """Schema-related error."""


class SchemaInferenceError(SchemaError):
    """Could not infer schema information."""


class SchemaUpdateError(SchemaError):
    """Failed to update schema."""


class JoinInferenceError(SchemaError):
    """Could not infer valid join keys."""


# =========================
# LLM
# =========================

class LLMError(FeatureSelectionError):
    """Base LLM error."""


class LLMInvocationError(LLMError):
    """LLM call failed."""


class LLMEmptyResponseError(LLMError):
    """LLM returned empty response."""


class LLMResponseFormatError(LLMError):
    """LLM returned unexpected format."""


class LLMInvalidJSONError(LLMResponseFormatError):
    """LLM returned invalid JSON."""


class LLMToolCallError(LLMError):
    """LLM produced invalid tool call / tool set."""


# =========================
# Features / evaluation
# =========================

class FeatureError(FeatureSelectionError):
    """Base feature-related error."""


class FeatureGenerationError(FeatureError):
    """Feature generation failed."""


class FeatureValidationError(FeatureError):
    """Generated features are invalid."""


class FeatureEvaluationError(FeatureError):
    """Feature evaluation failed."""


class ExperimentMemoryError(FeatureError):
    """Experiment registry / memory operation failed."""


# =========================
# Output / submission
# =========================

class OutputError(FeatureSelectionError):
    """Output file generation error."""


class SubmissionFormatError(OutputError):
    """Submission format is invalid."""

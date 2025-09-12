"""Custom exceptions for NiMARE."""


class NiMAREError(Exception):
    """Base class for NiMARE exceptions."""


class InvalidStudysetError(NiMAREError):
    """Exception raised when a studyset is invalid or malformed."""


class ConversionWarning(UserWarning):
    """Warning raised during data conversion processes."""

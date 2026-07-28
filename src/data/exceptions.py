"""Custom exceptions for the BraTS data-reading module.

Using specific exception types (instead of generic ``ValueError`` /
``FileNotFoundError``) lets calling code distinguish dataset-level problems
from programming errors, and makes failures self-explanatory in logs and
tracebacks.
"""

from __future__ import annotations


class BraTSDatasetError(Exception):
    """Base class for all errors raised by the BraTS reading module."""


class DatasetRootNotFoundError(BraTSDatasetError):
    """Raised when the configured dataset root directory does not exist."""


class PatientNotFoundError(BraTSDatasetError):
    """Raised when a requested patient ID does not exist in the dataset."""


class IncompletePatientError(BraTSDatasetError):
    """Raised when trying to load data for a patient that failed validation."""

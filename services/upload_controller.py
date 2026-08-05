"""Uploaded-file lifecycle from validation through disk-backed storage."""

from dataclasses import dataclass
import hashlib

from modules.large_data_engine import create_dataset_store
from services.file_service import (
    get_excel_sheet_names,
    load_dataframe,
    validate_uploaded_file,
)
from services.session_service import reset_for_dataset


@dataclass(frozen=True)
class UploadCandidate:
    original_name: str
    safe_name: str
    extension: str
    file_bytes: bytes
    sheet_names: list | None


@dataclass(frozen=True)
class LoadedDataset:
    dataframe: object
    dataset_store: object
    file_id: tuple
    safe_name: str
    selected_sheet: str | None


def inspect_uploaded_file(uploaded_file):
    """Read and validate an uploaded file once before any parsing occurs."""
    file_bytes = uploaded_file.getvalue()
    safe_name, extension = validate_uploaded_file(file_bytes, uploaded_file.name)
    sheet_names = get_excel_sheet_names(file_bytes, safe_name)
    return UploadCandidate(
        original_name=uploaded_file.name,
        safe_name=safe_name,
        extension=extension,
        file_bytes=file_bytes,
        sheet_names=sheet_names,
    )


def load_uploaded_dataset(candidate, selected_sheet, state):
    """Load the selected sheet and create/reuse its Parquet-backed store."""
    digest = hashlib.sha256(candidate.file_bytes).hexdigest()
    file_id = (digest, selected_sheet)
    reset_for_dataset(state, file_id)

    if state["original_dataframe"] is None:
        state["original_dataframe"] = load_dataframe(
            candidate.file_bytes, candidate.safe_name.lower(), selected_sheet
        )
    dataframe = state["original_dataframe"]

    if state["dataset_store"] is None:
        state["dataset_store"] = create_dataset_store(
            dataframe=dataframe,
            file_bytes=candidate.file_bytes,
            source_name=candidate.safe_name,
            sheet_name=selected_sheet,
        )
    return LoadedDataset(
        dataframe=dataframe,
        dataset_store=state["dataset_store"],
        file_id=file_id,
        safe_name=candidate.safe_name,
        selected_sheet=selected_sheet,
    )

"""Validated multi-format dataset ingestion."""

from io import BytesIO
from pathlib import Path
import os
import tempfile

import pandas as pd


SUPPORTED_EXTENSIONS = [
    "csv", "tsv", "txt", "xlsx", "xls", "ods", "json", "parquet",
    "feather", "sav", "sas7bdat", "dta", "rdata", "rds", "rda", "html",
]
BLOCKED_EXTENSIONS = {"pkl", "pickle"}
MAX_FILE_SIZE_MB = 100


def validate_uploaded_file(file_bytes, file_name):
    safe_name = Path(file_name).name
    extension = Path(safe_name).suffix.lower().lstrip(".")
    if safe_name != file_name or not extension:
        raise ValueError("The uploaded file name or extension is invalid.")
    if extension in BLOCKED_EXTENSIONS:
        raise ValueError("Pickle files are not supported because they can contain executable code.")
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("This file type is not supported.")
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(file_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise ValueError(f"File exceeds the {MAX_FILE_SIZE_MB} MB safety limit.")
    signatures = {
        "parquet": (b"PAR1",), "feather": (b"ARROW1",),
        "xlsx": (b"PK\x03\x04",), "ods": (b"PK\x03\x04",),
        "xls": (b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",),
    }
    expected = signatures.get(extension)
    if expected and not any(file_bytes.startswith(sig) for sig in expected):
        raise ValueError("The file content does not match its extension. The file may be damaged or incorrectly renamed.")
    return safe_name, extension


def get_excel_sheet_names(file_bytes, file_name):
    if file_name.lower().endswith((".xlsx", ".xls")):
        return pd.ExcelFile(BytesIO(file_bytes)).sheet_names
    return None


def _detect_delimiter(file_bytes):
    sample = file_bytes[:5000].decode("utf-8", errors="ignore")
    counts = {",": sample.count(","), ";": sample.count(";"), "\t": sample.count("\t"), "|": sample.count("|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _first_dataframe_from_dict(result_dict):
    if not result_dict:
        raise ValueError("No data frames found inside the R file.")
    return list(result_dict.values())[0]


def _read_with_temporary_file(file_bytes, suffix, reader):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return reader(tmp_path)
    finally:
        os.unlink(tmp_path)


def load_dataframe(file_bytes, file_name, sheet_name=None):
    name = file_name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(BytesIO(file_bytes))
    if name.endswith(".tsv"):
        return pd.read_csv(BytesIO(file_bytes), sep="\t")
    if name.endswith(".txt"):
        return pd.read_csv(BytesIO(file_bytes), sep=_detect_delimiter(file_bytes), engine="python")
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)
    if name.endswith(".ods"):
        return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name, engine="odf")
    if name.endswith(".json"):
        try:
            return pd.read_json(BytesIO(file_bytes))
        except ValueError:
            return pd.read_json(BytesIO(file_bytes), lines=True)
    if name.endswith(".parquet"):
        return pd.read_parquet(BytesIO(file_bytes))
    if name.endswith(".feather"):
        return pd.read_feather(BytesIO(file_bytes))
    if name.endswith(".sav"):
        import pyreadstat
        return _read_with_temporary_file(file_bytes, ".sav", lambda path: pyreadstat.read_sav(path)[0])
    if name.endswith(".sas7bdat"):
        import pyreadstat
        return _read_with_temporary_file(file_bytes, ".sas7bdat", lambda path: pyreadstat.read_sas7bdat(path)[0])
    if name.endswith(".dta"):
        return pd.read_stata(BytesIO(file_bytes))
    if name.endswith((".rdata", ".rds", ".rda")):
        import pyreadr
        suffix = ".rds" if name.endswith(".rds") else ".RData"
        result = _read_with_temporary_file(file_bytes, suffix, pyreadr.read_r)
        return _first_dataframe_from_dict(result)
    if name.endswith(".html"):
        tables = pd.read_html(BytesIO(file_bytes))
        if not tables:
            raise ValueError("No tables found in the HTML file.")
        return tables[0]
    raise ValueError("Unsupported file type: " + Path(file_name).suffix)

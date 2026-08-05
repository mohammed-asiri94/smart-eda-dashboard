"""Core ingestion, storage, session, and full-data regression tests."""

import ast
from pathlib import Path
import sys
import types
import unittest

import pandas as pd

from modules.large_data_engine import create_dataset_store
from services.file_service import validate_uploaded_file
from services.session_service import initialize_session_state, reset_for_dataset
from services.upload_controller import inspect_uploaded_file, load_uploaded_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeUpload:
    def __init__(self, name, content):
        self.name = name
        self._content = content

    def getvalue(self):
        return self._content


class CoreServiceTests(unittest.TestCase):
    def test_validation_rejects_empty_and_pickle_files(self):
        with self.assertRaises(ValueError):
            validate_uploaded_file(b"", "empty.csv")
        with self.assertRaises(ValueError):
            validate_uploaded_file(b"unsafe", "model.pkl")

    def test_session_results_survive_same_file_and_reset_for_new_file(self):
        state = {}
        initialize_session_state(state)
        self.assertTrue(reset_for_dataset(state, ("file-a", None)))
        state["overview_summary"] = "cached"
        self.assertFalse(reset_for_dataset(state, ("file-a", None)))
        self.assertEqual(state["overview_summary"], "cached")
        self.assertTrue(reset_for_dataset(state, ("file-b", None)))
        self.assertIsNone(state["overview_summary"])

    def test_upload_lifecycle_creates_and_reuses_parquet_store(self):
        state = {}
        initialize_session_state(state)
        candidate = inspect_uploaded_file(
            FakeUpload("sample.csv", b"x,group\n1,a\n2,b\n")
        )
        first = load_uploaded_dataset(candidate, None, state)
        self.assertEqual(first.dataframe.shape, (2, 2))
        self.assertTrue(first.dataset_store.parquet_path.exists())
        state["overview_summary"] = "cached"
        second = load_uploaded_dataset(candidate, None, state)
        self.assertEqual(first.file_id, second.file_id)
        self.assertEqual(state["overview_summary"], "cached")

    def test_large_store_keeps_all_500000_rows(self):
        row_count = 500_000
        dataframe = pd.DataFrame(
            {
                "row_id": range(row_count),
                "value": [float(i % 101) for i in range(row_count)],
            }
        )
        store = create_dataset_store(
            dataframe=dataframe,
            file_bytes=b"permanent-large-data-regression-test-v1",
            source_name="large.csv",
        )
        self.assertTrue(store.is_large)
        self.assertEqual(store.row_count, row_count)
        self.assertEqual(store.core_metrics(include_duplicates=False)["rows"], row_count)
        self.assertEqual(len(store.read_dataframe(columns=["row_id"])), row_count)


class ArchitectureSafetyTests(unittest.TestCase):
    def test_model_modules_do_not_call_dataframe_sample(self):
        violations = []
        for path in (PROJECT_ROOT / "modules" / "models").glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "sample"
                ):
                    violations.append(f"{path.name}:{node.lineno}")
        self.assertEqual(violations, [], "Model sampling found: " + ", ".join(violations))

    def test_clean_dataset_calls_are_guarded_by_a_button(self):
        path = PROJECT_ROOT / "ui" / "cleaning_page.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        guarded_calls = 0

        def is_button_condition(test):
            return any(
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "button"
                for node in ast.walk(test)
            )

        def walk_statements(statements, button_guard=False):
            nonlocal guarded_calls
            for statement in statements:
                guarded = button_guard
                if isinstance(statement, ast.If):
                    guarded = guarded or is_button_condition(statement.test)
                    walk_statements(statement.body, guarded)
                    walk_statements(statement.orelse, button_guard)
                    continue
                for node in ast.walk(statement):
                    if (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == "clean_dataset"
                    ):
                        self.assertTrue(guarded, f"Unguarded cleaning call at line {node.lineno}")
                        guarded_calls += 1

        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "render_cleaning_page"
        )
        walk_statements(function.body)
        self.assertGreater(guarded_calls, 0)


if __name__ == "__main__":
    unittest.main()

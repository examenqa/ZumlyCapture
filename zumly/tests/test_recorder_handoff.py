"""Tests for the recorder-to-tray result payload contract."""

import json
import os

from zumly_capture.identity import RUNTIME_DIRECTORY_NAME
from zumly.main import (
    _capture_project_bridge_path,
    _write_capture_project_bridge,
    _write_result_payload,
)


def test_result_payload_is_published_atomically(tmp_path) -> None:
    result_path = tmp_path / "result.json"
    payload = {
        "status": "success",
        "projectPath": str(tmp_path / "project.json"),
        "returnCode": 0,
    }

    assert _write_result_payload(str(result_path), payload) is True
    assert json.loads(result_path.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.glob("*.tmp")) == []


def test_capture_project_bridge_uses_runtime_directory_and_writes_atomically(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("zumly.main.tempfile.gettempdir", lambda: str(tmp_path))
    path = _capture_project_bridge_path(1234.0, "abc-def")

    assert os.path.commonpath(
        (path, str(tmp_path / RUNTIME_DIRECTORY_NAME / "bridges"))
    ) == str(
        tmp_path / RUNTIME_DIRECTORY_NAME / "bridges"
    )
    assert os.path.basename(path) == "capture_1234_abcdef_project.json"

    _write_capture_project_bridge(path, {"session": "abc"})
    assert json.loads(open(path, encoding="utf-8").read()) == {"session": "abc"}
    assert list((tmp_path / "Zumly" / "bridges").glob("*.tmp")) == []

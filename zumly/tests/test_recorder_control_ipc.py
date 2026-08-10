import json

from main import _read_control_payload


def test_control_ipc_ignores_stale_sequences(tmp_path) -> None:
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({"sequence": 3, "action": "pause"}),
        encoding="utf-8",
    )

    assert _read_control_payload(str(control), 2) == (3, "pause")
    assert _read_control_payload(str(control), 3) == (3, "")


def test_control_ipc_rejects_unknown_actions(tmp_path) -> None:
    control = tmp_path / "control.json"
    control.write_text(
        json.dumps({"sequence": 4, "action": "rewind"}),
        encoding="utf-8",
    )

    assert _read_control_payload(str(control), 1) == (1, "")

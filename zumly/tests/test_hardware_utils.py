"""Tests for the Windows GPU vendor probe and encoder mapping."""

from app import hardware_utils


def test_gpu_names_map_to_hardware_encoders(monkeypatch) -> None:
    monkeypatch.setattr(
        hardware_utils.subprocess,
        "check_output",
        lambda command, text, timeout: "Name\nNVIDIA GeForce RTX\nIntel UHD Graphics\n",
    )

    assert hardware_utils.detect_gpu_manufacturers() == {"nvidia", "intel"}
    monkeypatch.setattr(
        hardware_utils,
        "detect_available_encoders",
        lambda: [hardware_utils.ENCODER_NVIDIA, hardware_utils.ENCODER_INTEL, hardware_utils.ENCODER_SOFTWARE],
    )
    assert hardware_utils.detect_supported_hardware_encoders() == {
        hardware_utils.ENCODER_NVIDIA,
        hardware_utils.ENCODER_INTEL,
    }


def test_gpu_probe_failure_leaves_software_as_only_fallback(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise FileNotFoundError("wmic")

    monkeypatch.setattr(hardware_utils.subprocess, "check_output", fail)
    monkeypatch.setattr(hardware_utils, "detect_available_encoders", lambda: [hardware_utils.ENCODER_SOFTWARE])

    assert hardware_utils.detect_supported_hardware_encoders() == set()


def test_gpu_probe_uses_powershell_when_wmic_is_missing(monkeypatch) -> None:
    calls = []

    def query(command, text, timeout):
        calls.append(command[0])
        if command[0] == "wmic":
            raise FileNotFoundError("wmic")
        return "Name\nNVIDIA GeForce RTX 3050\n"

    monkeypatch.setattr(hardware_utils.subprocess, "check_output", query)

    assert hardware_utils.detect_gpu_manufacturers() == {"nvidia"}
    assert calls == ["wmic", "powershell.exe"]

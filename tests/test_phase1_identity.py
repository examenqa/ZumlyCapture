from __future__ import annotations

from uuid import UUID

from zumly.app.single_instance import ZumlyCaptureSingleInstance
from zumly_capture import APP_IDENTITY, __version__


def test_standalone_identity_is_distinct_from_zumly() -> None:
    assert APP_IDENTITY.product_name == "Zumly Capture"
    assert APP_IDENTITY.application_id == "com.zumly.capture"
    assert APP_IDENTITY.settings_application_name != "Zumly"
    assert APP_IDENTITY.executable_name == "ZumlyCapture"


def test_single_instance_defaults_use_capture_namespace() -> None:
    guard = ZumlyCaptureSingleInstance(kernel32=object())

    assert guard.mutex_name.startswith(r"Local\ZumlyCapture.Tray.")
    assert guard.activation_event_name.startswith(
        r"Local\ZumlyCapture.Tray.Activate."
    )


def test_installer_identity_is_a_valid_uuid() -> None:
    assert UUID(APP_IDENTITY.installer_app_id.strip("{}"))


def test_standalone_version_starts_a_fresh_release_line() -> None:
    assert __version__ == "0.5.2"

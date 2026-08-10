"""Native single-instance ownership tests for the tray application."""

from __future__ import annotations

import os
import uuid

import pytest

from app.single_instance import ZumlySingleInstance


pytestmark = pytest.mark.skipif(os.name != "nt", reason="Windows named objects are required")


def test_second_tray_process_is_rejected_and_activates_primary() -> None:
    suffix = uuid.uuid4().hex
    mutex_name = f"Local\\Zumly.Test.{suffix}"
    event_name = f"Local\\Zumly.Test.Activate.{suffix}"
    primary = ZumlySingleInstance(
        mutex_name=mutex_name,
        activation_event_name=event_name,
    )
    secondary = ZumlySingleInstance(
        mutex_name=mutex_name,
        activation_event_name=event_name,
    )

    try:
        assert primary.acquire() is True
        assert secondary.acquire() is False
        assert secondary.signal_primary(timeout_s=0.25) is True
        assert primary.consume_activation() is True
        assert primary.consume_activation() is False
    finally:
        secondary.close()
        primary.close()

    replacement = ZumlySingleInstance(
        mutex_name=mutex_name,
        activation_event_name=event_name,
    )
    try:
        assert replacement.acquire() is True
    finally:
        replacement.close()

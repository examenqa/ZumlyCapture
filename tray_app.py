"""Zumly Capture tray entry point.

The Qt tray UI is loaded only for the normal tray process. The frozen
``--headless-engine`` path imports only the capture worker, preserving the
capture process boundary.
"""

from __future__ import annotations

import logging
import sys

from zumly_capture import __version__
from zumly_capture.identity import (
    APPLICATION_NAME,
    ORGANIZATION_DOMAIN,
    ORGANIZATION_NAME,
    PRODUCT_NAME,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [TRAY] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def entry() -> int:
    if "--headless-engine" in sys.argv:
        sys.argv.remove("--headless-engine")
        from zumly.main import main as headless_main

        return int(headless_main() or 0)

    from PySide6.QtWidgets import QApplication
    from zumly.app.icon_loader import get_brand_icon
    from zumly.app.qt_tray import QtZumlyCaptureTray
    from zumly.app.single_instance import ZumlyCaptureSingleInstance

    instance = ZumlyCaptureSingleInstance()
    if not instance.acquire():
        instance.signal_primary()
        return 0

    app = QApplication(sys.argv)
    app.setApplicationName(APPLICATION_NAME)
    app.setApplicationDisplayName(PRODUCT_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(ORGANIZATION_NAME)
    app.setOrganizationDomain(ORGANIZATION_DOMAIN)
    app.setWindowIcon(get_brand_icon())
    app.setQuitOnLastWindowClosed(False)
    app.aboutToQuit.connect(instance.close)
    tray = QtZumlyCaptureTray(app, instance_guard=instance)
    tray.run()
    return app.exec()


def main() -> int:
    return entry()


if __name__ == "__main__":
    raise SystemExit(entry())

# -*- mode: python ; coding: utf-8 -*-

from zumly_capture.identity import EXECUTABLE_NAME


brand_icon = "zumly/app/branding/generated/zumly.ico"
app_datas = [
    ("zumly/app/branding/*.svg", "zumly/app/branding"),
    ("zumly/app/branding/generated/*", "zumly/app/branding/generated"),
    ("zumly/app/cursors/*.svg", "zumly/app/cursors"),
    ("zumly/app/fonts/*", "zumly/app/fonts"),
    ("zumly/app/icons/*.svg", "zumly/app/icons"),
]

analysis = Analysis(
    ["tray_app.py"],
    pathex=[".", "zumly"],
    binaries=[],
    datas=app_datas,
    hiddenimports=[
        "zumly_capture",
        "zumly_capture.audio",
        "zumly_capture.capture_ui",
        "zumly_capture.session",
        "zumly_capture.settings",
        "zumly_capture.settings_dialog",
        "zumly_capture.screenshot",
        "zumly_capture.wgc",
        "windows_capture.windows_capture",
        "zumly.main",
        "zumly.app.qt_tray",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "PySide6.QtSvg",
        "PySide6.QtWidgets",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Product artifacts must not gain OpenCV or NumPy as mandatory runtime
    # dependencies. Copied seed paths that still import NumPy remain
    # transitional extraction work.
    excludes=["cv2", "numpy"],
    noarchive=False,
)

pyz = PYZ(analysis.pure)

exe = EXE(
    pyz,
    analysis.scripts,
    [],
    exclude_binaries=True,
    name=EXECUTABLE_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    icon=brand_icon,
)

coll = COLLECT(
    exe,
    analysis.binaries,
    analysis.datas,
    strip=False,
    upx=True,
    name="zumly-capture",
)

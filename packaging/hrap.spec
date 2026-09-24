# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the HRAP (HCAT Fork) desktop app.

Versioned artifacts are produced by packaging/build_windows.py so later
releases only need a version bump + tag (vX.Y.Z).
"""
from pathlib import Path

from PyInstaller.utils.hooks import collect_all, collect_data_files

ROOT = Path(SPECPATH).resolve().parent
SRC = ROOT / "src"

datas = collect_data_files("hrap")
binaries = []
hiddenimports = [
    "scipy.optimize",
    "scipy.optimize._zeros",
    "hrap",
    "hrap.gui.main",
    "hrap.gui.theme",
    "hrap.gui.viz",
]

for pkg in ("PySide6", "pyqtgraph", "CoolProp"):
    pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
    datas += pkg_datas
    binaries += pkg_binaries
    hiddenimports += pkg_hidden

_SKIP = (
    "qt6webengine",
    "qt6quick",
    "qt6qml",
    "qt6multimedia",
    "qt6pdf",
    "qt6designer",
    "qt6charts",
    "qt6datavisualization",
    "qt6remoteobjects",
    "qt6sensors",
    "qt6bluetooth",
    "qt6nfc",
    "qt63d",
    "qt6quick3d",
    "linguist",
    "designer.exe",
    "assistant.exe",
)


def _keep(item) -> bool:
    name = str(item[0] if isinstance(item, (list, tuple)) else item).lower()
    return not any(token in name for token in _SKIP)


datas = [item for item in datas if _keep(item)]
binaries = [item for item in binaries if _keep(item)]

a = Analysis(
    [str(SRC / "hrap" / "__main__.py")],
    pathex=[str(SRC)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(Path(SPECPATH) / "pyi_rth_hrap.py")],
    excludes=["pytest", "IPython", "matplotlib", "tkinter"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="HRAP",
    debug=False,
    icon=str(SRC / "hrap" / "resources" / "icon.ico"),
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="HRAP",
)

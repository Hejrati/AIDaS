# -*- mode: python ; coding: utf-8 -*-

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH)

datas = [
    (str(project_root / "assets"), "assets"),
    (str(project_root / "OCT Segmenter" / "config.json"), "OCT Segmenter"),
    (str(project_root / "OCT Segmenter" / "Model" / "human_OCT.h5"), "OCT Segmenter/Model"),
    (str(project_root / "OCT Segmenter" / "Model" / "model_config.json"), "OCT Segmenter/Model"),
    (
        str(project_root / "OCT Segmenter" / "AI_ForAIDAS" / "model_img.pth"),
        "OCT Segmenter/AI_ForAIDAS",
    ),
    (
        str(project_root / "OCT Segmenter" / "AI_ForAIDAS" / "vline_model.pth"),
        "OCT Segmenter/AI_ForAIDAS",
    ),
]
binaries = []
hiddenimports = [
    "aidas.ai_for_aidas_cli",
    "aidas.ai_for_aidas_inference",
    "torch",
    "torch.nn",
    "torch.nn.functional",
]


def collect_package(package_name, required=True):
    if importlib.util.find_spec(package_name) is None:
        if required:
            raise RuntimeError(
                f"Cannot bundle {package_name!r}; install the project requirements before running PyInstaller."
            )
        return

    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas.extend(package_datas)
    binaries.extend(package_binaries)
    hiddenimports.extend(package_hiddenimports)


for package_name in (
    "torch",
    "numpy",
    "scipy",
    "PIL",
    "matplotlib",
):
    collect_package(package_name)


for package_name in (
    "pyreadr",
    "xarray",
    "imgviz",
    "prettytable",
    "art",
    "cmap",
):
    collect_package(package_name, required=False)


a = Analysis(
    ['run_aidas.py'],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AIDaS',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(project_root / "assets" / "aidas.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='AIDaS',
)

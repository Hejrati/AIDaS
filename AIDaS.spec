# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path


PROJECT_ROOT = Path(globals().get('SPECPATH', Path.cwd())).resolve()
ICON_PATH = PROJECT_ROOT / 'assets' / 'aidas.ico'
PNG_ICON_PATH = PROJECT_ROOT / 'assets' / 'aidas.png'

DATA_FILES = []
if ICON_PATH.is_file():
    DATA_FILES.append((str(ICON_PATH), 'assets'))
if PNG_ICON_PATH.is_file():
    DATA_FILES.append((str(PNG_ICON_PATH), 'assets'))


# Ensure EXE icon is always set to your ICO when present.
ICON_ARG = str(ICON_PATH) if ICON_PATH.is_file() else None


a = Analysis(
    ['run_aidas.py'],
    pathex=[],
    binaries=[],
    datas=DATA_FILES,
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['oct_image_segmentation_models', 'oct_segmenter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='AIDaS',
    icon=ICON_ARG,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

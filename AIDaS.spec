# -*- mode: python ; coding: utf-8 -*-

import importlib.util
from pathlib import Path

from PyInstaller.utils.hooks import collect_dynamic_libs


project_root = Path(SPECPATH)

datas = [
    (str(project_root / "assets"), "assets"),
    (
        str(project_root / "OCT Segmenter" / "AI_ForAIDAS" / "model_img.onnx"),
        "OCT Segmenter/AI_ForAIDAS",
    ),
]
binaries = collect_dynamic_libs("pyreadr")
hiddenimports = [
    "aidas.ai.worker",
    "aidas.ai.client",
    "aidas.ai.inference",
    "onnxruntime",
    "onnxruntime.capi._pybind_state",
]


def collect_r_scripts():
    """Bundle only the release R scripts stored at the project root."""
    script_paths = sorted(
        (path for path in project_root.iterdir() if path.is_file() and path.suffix.lower() == ".r"),
        key=lambda path: path.name.lower(),
    )

    if not script_paths:
        raise RuntimeError("Cannot build AIDaS because no R scripts were found.")

    for source_path in script_paths:
        datas.append((str(source_path), "."))


collect_r_scripts()


required_runtime_imports = (
    "numpy",
    "scipy",
    "PIL",
    "matplotlib",
    "pyreadr",
    "packaging",
    "onnxruntime",
)
missing_runtime_imports = [
    package_name
    for package_name in required_runtime_imports
    if importlib.util.find_spec(package_name) is None
]
if missing_runtime_imports:
    raise RuntimeError(
        "Cannot build AIDaS because required runtime packages are missing: "
        f"{', '.join(missing_runtime_imports)}. Install requirements.txt first."
    )


a = Analysis(
    ['run_aidas.py'],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={"matplotlib": {"backends": ["TkAgg"]}},
    runtime_hooks=[],
    excludes=[
        "torch",
        "tensorflow",
        "onnx",
        "psutil",
        "pytest",
        "_pytest",
        "IPython",
        "dask",
        "sphinx",
        "sphinxcontrib",
        "docutils",
        "jedi",
        "parso",
        "pyarrow",
        "numba",
        "llvmlite",
        "fsspec",
        "jinja2",
        "pygments",
        "chardet",
        "win32com",
        "pythoncom",
        "pywintypes",
        "botocore",
        "boto3",
        "s3fs",
    ],
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

# -*- mode: python ; coding: utf-8 -*-

import importlib.util
from pathlib import Path

from PyInstaller.building.splash import Splash as PyInstallerSplash
from PyInstaller.utils.hooks import collect_all


project_root = Path(SPECPATH)

datas = [
    (str(project_root / "assets"), "assets"),
    (
        str(project_root / "OCT Segmenter" / "AI_ForAIDAS" / "model_img.pth"),
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


class AIDaSSplash(PyInstallerSplash):
    """PyInstaller splash with AIDaS identity and no Tk taskbar entry."""

    def generate_script(self):
        script = super().generate_script()
        identity_script = (
            'wm title . "AIDaS"\n'
            'if {$tcl_platform(platform) eq "windows"} { wm attributes . -toolwindow 1 }\n'
            'wm iconphoto . -default splash_image\n'
        )
        script = script.replace("raise .", identity_script + "raise .")
        Path(self.script_name).write_text(script, encoding="utf-8")
        return script


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

splash = AIDaSSplash(
    str(project_root / "assets" / "startup_splash.png"),
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    max_img_size=(572, 816),
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    splash.binaries,
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

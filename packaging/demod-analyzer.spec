# PyInstaller spec: three executables, one shared copy of Python/Qt/numpy.
#
# Each tool gets its own Analysis (so its own entry point and its own icon /
# console setting), but all three EXEs are handed to a single COLLECT.  COLLECT
# writes one `_internal` directory next to the three .exe files and every
# executable loads its runtime from there, so python3*.dll, the Qt6 DLLs,
# numpy and scipy exist exactly once in the bundle.
#
# The reference captures in recordings/ are copied in after COLLECT, so they
# land next to the .exe files rather than inside _internal where PyInstaller
# puts declared `datas`.
#
#     pyinstaller packaging/demod-analyzer.spec       (run from the repo root)

import os
import shutil

# Run from the repo root; SPECPATH is packaging/, the sources are one level up.
ROOT = os.path.abspath(os.path.join(SPECPATH, os.pardir))

# console=False hides the terminal window for the two GUI-only tools.
# make_test_signal is also a command line generator, so it keeps its console.
APPS = [
    ("demod_analyzer",  "demod_analyzer.py",  False),
    ("iq_splitter",     "iq_splitter.py",     False),
    ("make_test_signal", "make_test_signal.py", True),
]

# Qt ships far more than pyqtgraph uses; WebEngine alone is >100 MB.  None of
# these are imported by the three tools, and excluding them keeps the shared
# _internal directory to the parts that are actually loaded.
EXCLUDES = [
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick",
    "PySide6.QtQuick", "PySide6.QtQuickWidgets", "PySide6.QtQml",
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets",
    "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtBluetooth",
    "PySide6.QtNetworkAuth", "PySide6.QtPositioning", "PySide6.QtSerialPort",
    "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtWebSockets", "PySide6.QtWebChannel",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DAnimation",
    "PySide6.Qt3DExtras", "PySide6.Qt3DInput", "PySide6.Qt3DLogic",
    "tkinter", "matplotlib", "IPython", "pytest", "PIL",
]

analyses, executables = [], []

for name, script, console in APPS:
    a = Analysis(
        [os.path.join(ROOT, script)],
        pathex=[ROOT],
        binaries=[],
        datas=[],
        hiddenimports=[],
        hookspath=[],
        runtime_hooks=[],
        excludes=EXCLUDES,
        noarchive=False,
    )
    analyses.append(a)
    executables.append(
        EXE(
            PYZ(a.pure),
            a.scripts,
            [],
            exclude_binaries=True,      # everything shared lands in COLLECT
            name=name,
            console=console,
            debug=False,
            strip=False,
            upx=False,
        )
    )

# One COLLECT for all three.  Identical files contributed by more than one
# Analysis are written once; PyInstaller warns about the duplicates and keeps a
# single copy.
collected = []
for a in analyses:
    collected += [a.binaries, a.datas]

BUNDLE_NAME = "demod-analyzer"

COLLECT(
    *executables,
    *collected,
    strip=False,
    upx=False,
    name=BUNDLE_NAME,
)

# A spec is executed top to bottom, so COLLECT has finished writing the bundle
# by the time this runs. Plain copy rather than Analysis(datas=...) because the
# captures are files the user opens from the analyzer, not resources the code
# loads -- they belong beside the executables, not under _internal.
shutil.copytree(
    os.path.join(ROOT, "recordings"),
    os.path.join(DISTPATH, BUNDLE_NAME, "recordings"),
    dirs_exist_ok=True,
)

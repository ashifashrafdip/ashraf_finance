# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

datas = [
    ('objects/meanshape_68.pkl', 'objects'),
    ('objects/meanshape_68.pkl', 'insightface/data/objects'),
    ('kyc_verifier.ico', '.'),
]
binaries = []
hiddenimports = [
    'PySide6',
    'PySide6.QtCore',
    'PySide6.QtGui',
    'PySide6.QtWidgets',
    'biometrics_engine',
    'PIL',
    'PIL.Image',
    'PIL.ImageOps',
    'PIL.ExifTags',
    'cv2',
    'numpy',
    'scipy',
    'skimage',
    'skimage.transform',
    'insightface',
    'onnxruntime',
]

d_ort, b_ort, h_ort = collect_all('onnxruntime')
datas += d_ort
binaries += b_ort
hiddenimports += h_ort

d_if, b_if, h_if = collect_all('insightface')
datas += d_if
binaries += b_if
hiddenimports += h_if

d_pyside, b_pyside, h_pyside = collect_all('PySide6')
datas += d_pyside
binaries += b_pyside
hiddenimports += h_pyside

block_cipher = None

a = Analysis(
    ['gui_verifier.py'],
    pathex=['.'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['tkinter', 'matplotlib', 'torch', 'torchaudio', 'torchvision'],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='KYC_Verifier',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='kyc_verifier.ico',
)

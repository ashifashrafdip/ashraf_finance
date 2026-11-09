# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

datas = [
    ('static/index.html', 'static'),
    ('objects/meanshape_68.pkl', 'objects'),
    ('objects/meanshape_68.pkl', 'insightface/data/objects'),
]
binaries = []
hiddenimports = [
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.loops',
    'uvicorn.loops.auto',
    'uvicorn.protocols',
    'uvicorn.protocols.http',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.websockets',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan',
    'uvicorn.lifespan.on',
    'fastapi',
    'multipart',
    'python_multipart',
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

# Collect onnxruntime & insightface
d_ort, b_ort, h_ort = collect_all('onnxruntime')
datas += d_ort
binaries += b_ort
hiddenimports += h_ort

d_if, b_if, h_if = collect_all('insightface')
datas += d_if
binaries += b_if
hiddenimports += h_if

block_cipher = None

a = Analysis(
    ['kyc_verifier.py'],
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

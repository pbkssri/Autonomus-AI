# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for Autonomous AI Agent
Build Windows executable with: pyinstaller autonomous_agent.spec
"""

import os
import sys

block_cipher = None

# Project directory - use current working directory
project_dir = os.getcwd()

a = Analysis(
    ['main.py'],
    pathex=[project_dir],
    binaries=[],
    datas=[
        # Include config directory
        (os.path.join(project_dir, 'config'), 'config'),
        # Include data directory
        (os.path.join(project_dir, 'data'), 'data'),
        # Include workspace directory
        (os.path.join(project_dir, 'workspace'), 'workspace'),
    ],
    hiddenimports=[
        'aiohttp',
        'aiohttp.abc',
        'aiohttp.base_protocol',
        'aiohttp.client',
        'aiohttp.client_proto',
        'aiohttp.client_reqrep',
        'aiohttp.client_ws',
        'aiohttp.connector',
        'aiohttp.cookiejar',
        'aiohttp.formdata',
        'aiohttp.hdrs',
        'aiohttp.http_exceptions',
        'aiohttp.http_parser',
        'aiohttp.http_protocol',
        'aiohttp.http_websocket',
        'aiohttp.workers',
        'asyncio',
        'json',
        'logging',
        'configparser',
        'sqlite3',
        'tkinter',
        'tkinter.ttk',
        'tkinter.scrolledtext',
        'threading',
        'concurrent.futures',
        'pathlib',
        'hashlib',
        'uuid',
        'datetime',
        'tempfile',
        'shutil',
        'subprocess',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='AutonomousAI-Agent',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True for debug mode
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path if available
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='AutonomousAI-Agent',
)

@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Voice Splitter
set HF_HUB_DISABLE_SYMLINKS_WARNING=1

if exist ".venv\Scripts\python.exe" goto run

echo.
echo  ============================================================
echo   Voice Splitter: first-time setup
echo  ============================================================
echo   This installs what it needs into this folder (about 3 GB with
echo   GPU support, 1 GB without). It only happens once.
echo.

rem ---- 1. Python 3.10-3.12 (installs 3.11 with winget if none is found)
set "PY="
for %%V in (3.11 3.12 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
  )
)
if not defined PY (
  python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo  [1/4] Python 3.11 isn't installed. Installing it now...
  where winget >nul 2>nul
  if errorlevel 1 (
    echo.
    echo  Couldn't install Python automatically. Install Python 3.11 from
    echo  https://www.python.org/downloads/ ^(tick "Add python.exe to PATH"^),
    echo  then double-click this file again.
    pause & exit /b 1
  )
  winget install -e --id Python.Python.3.11 --scope user --accept-package-agreements --accept-source-agreements
  set "PY=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
  if not exist "!PY!" (
    echo  Python was installed but this window can't see it yet. Close it and
    echo  double-click Start Voice Splitter.bat again.
    pause & exit /b 1
  )
  set "PY="!PY!""
) else (
  echo  [1/4] Found Python: !PY!
)

echo  [2/4] Creating the app's own Python environment...
!PY! -m venv .venv
if not exist ".venv\Scripts\python.exe" (
  echo  Couldn't create the environment. See README.md, "Troubleshooting".
  pause & exit /b 1
)
".venv\Scripts\python" -m pip install --upgrade pip --quiet

rem ---- 3. PyTorch: GPU build if there's an NVIDIA card, otherwise the smaller CPU build
where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo  [3/4] No NVIDIA graphics card found: installing PyTorch for the processor...
  set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
) else (
  echo  [3/4] NVIDIA graphics card found: installing PyTorch with GPU support...
  set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
)
".venv\Scripts\python" -m pip install torch torchaudio --index-url !TORCH_INDEX!
if errorlevel 1 goto failed

echo  [4/4] Installing Voice Splitter's parts...
".venv\Scripts\python" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo.
echo  Done. Opening Voice Splitter; it will walk you through the models.
echo.
goto run

:failed
echo.
echo  Setup didn't finish. Scroll up for the error, or see README.md "Troubleshooting".
echo  Fix it, then double-click this file again (it starts over cleanly).
rmdir /s /q .venv
pause & exit /b 1

:run
".venv\Scripts\python" voice_splitter.py %*
if errorlevel 1 pause

@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
if defined TENSORDOJO_PYTHON goto custom
if exist ".tensor-dojo-venv\Scripts\python.exe" goto venv
for %%V in (3.11 3.12 3.13 3.10 3) do (
  py -%%V -c "import sys,struct; assert sys.version_info >= (3,10) and struct.calcsize('P')==8" >nul 2>&1
  if not errorlevel 1 (
    set "PY_VERSION=%%V"
    goto pylauncher
  )
)
python -c "import sys,struct; assert sys.version_info >= (3,10) and struct.calcsize('P')==8" >nul 2>&1
if not errorlevel 1 goto plain
 echo Please install 64-bit Python 3.11 from https://www.python.org/downloads/
 echo This package does not bundle Python or torch binaries.
 pause
 exit /b 1
:custom
"%TENSORDOJO_PYTHON%" -u launch.py %*
goto finish
:venv
".tensor-dojo-venv\Scripts\python.exe" -u launch.py %*
goto finish
:pylauncher
py -%PY_VERSION% -u launch.py %*
goto finish
:plain
python -u launch.py %*
:finish
set "CODE=%errorlevel%"
if not "%CODE%"=="0" pause
exit /b %CODE%

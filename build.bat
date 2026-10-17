@echo off
rem Builds a standalone PhotoAssist.exe (GUI) into .\dist using PyInstaller.
rem Usage: build.bat [clean]
setlocal
cd /d "%~dp0"

set PY=python
if exist ".venv\Scripts\python.exe" set PY=.venv\Scripts\python.exe

if /i "%1"=="clean" (
    rmdir /s /q build dist 2>nul
    del PhotoAssist.spec 2>nul
)

%PY% -m pip install -q -r requirements-build.txt || goto :fail
%PY% -m PyInstaller --noconfirm --onefile --windowed --name PhotoAssist gui.py || goto :fail

echo Built: %~dp0dist\PhotoAssist.exe
exit /b 0

:fail
echo Build failed.
exit /b 1

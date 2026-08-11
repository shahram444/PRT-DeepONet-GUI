@echo off
REM ===========================================================================
REM  PRT-DeepONet Studio  -  double-click this file to start.
REM
REM  It looks for the project's own Python first (3D\.venv), because that is
REM  where torch and the rest are installed, and falls back to whatever python
REM  is on the PATH.
REM ===========================================================================
setlocal
set HERE=%~dp0
set ROOT=%HERE%..

if exist "%ROOT%\3D\.venv\Scripts\python.exe" (
    set PY="%ROOT%\3D\.venv\Scripts\python.exe"
) else if exist "%ROOT%\.venv\Scripts\python.exe" (
    set PY="%ROOT%\.venv\Scripts\python.exe"
) else (
    set PY=python
)

echo Starting PRT-DeepONet Studio with %PY%
%PY% "%HERE%prt_gui.py"

if errorlevel 1 (
    echo.
    echo ---------------------------------------------------------------
    echo  It did not start. The usual causes:
    echo.
    echo   1. Python is not installed, or not on the PATH.
    echo      Install it from python.org and tick "Add to PATH".
    echo.
    echo   2. A package is missing. Install them with:
    echo        %PY% -m pip install numpy scipy h5py matplotlib torch scikit-image
    echo.
    echo   3. Tk is missing. Reinstall Python and tick "tcl/tk and IDLE".
    echo ---------------------------------------------------------------
    pause
)
endlocal

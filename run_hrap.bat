@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title HRAP (HCAT Fork)

set "LOG=%~dp0hrap_launch.log"
> "%LOG%" echo %DATE% %TIME% starting HRAP launcher
>> "%LOG%" echo cwd=%CD%

rem Explorer double-click does not use the IDE PATH or env.
rem Never call WindowsApps\python.exe (Store stub). Never inherit offscreen Qt.
set "QT_QPA_PLATFORM="
if defined WINDIR (set "QT_QPA_FONTDIR=%WINDIR%\Fonts") else (set "QT_QPA_FONTDIR=C:\Windows\Fonts")
if not defined USERPROFILE set "USERPROFILE=%SystemDrive%\Users\%USERNAME%"
if not defined LOCALAPPDATA set "LOCALAPPDATA=%USERPROFILE%\AppData\Local"

set "PYTHON="
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PYTHON if exist "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe" set "PYTHON=%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" set "PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
if not defined PYTHON if exist "%LOCALAPPDATA%\Programs\Python\Launcher\py.exe" (
  for /f "usebackq delims=" %%I in (`"%LOCALAPPDATA%\Programs\Python\Launcher\py.exe" -3 -c "import sys; print(sys.executable)"`) do set "PYTHON=%%I"
)

if not defined PYTHON (
  >> "%LOG%" echo FAILED: no python.exe
  echo Python 3.10+ was not found.
  echo Install Python from python.org ^(not the Microsoft Store^) and retry.
  echo.
  pause
  exit /b 1
)

set "PYTHONW=%PYTHON%"
if /I "%PYTHON:~-10%"=="python.exe" set "PYTHONW=%PYTHON:~0,-10%pythonw.exe"
if not exist "%PYTHONW%" set "PYTHONW=%PYTHON%"

>> "%LOG%" echo python=%PYTHON%
>> "%LOG%" echo pythonw=%PYTHONW%

set "PYTHONPATH=%CD%\src;%PYTHONPATH%"

"%PYTHON%" -c "import hrap, PySide6, pyqtgraph, CoolProp" >> "%LOG%" 2>&1
if errorlevel 1 goto :install
goto :launch

:install
echo Installing HRAP and GUI dependencies...
>> "%LOG%" echo pip install
"%PYTHON%" -m pip install -e . >> "%LOG%" 2>&1
if errorlevel 1 goto :fail

:launch
>> "%LOG%" echo launching GUI
rem First quoted token after START is the window TITLE. An empty title is required
rem when the program path is quoted. Detach so closing the console does not kill HRAP.
start "" "%PYTHONW%" -m hrap
>> "%LOG%" echo start issued
exit /b 0

:fail
echo.
echo HRAP (HCAT Fork) failed to start.
echo Interpreter: %PYTHON%
echo Log: %LOG%
echo.
type "%LOG%"
echo.
pause
exit /b 1

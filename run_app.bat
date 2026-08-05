@echo off
cd /d "%~dp0"

where python >nul 2>&1
if %errorlevel%==0 goto run_python

py -3 -c "import sys" >nul 2>&1
if %errorlevel%==0 goto run_py

echo.
echo Python 3 is not installed or is not available on PATH.
echo Install Python 3.11 or 3.12 from https://www.python.org/downloads/
echo During installation, enable "Add Python to PATH".
echo Then run: python -m pip install -r requirements.txt
echo.
pause
exit /b 1

:run_python
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing project dependencies...
    python -m pip install -r requirements.txt
    if errorlevel 1 goto install_failed
)
python -m streamlit run app.py
goto end

:run_py
py -3 -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Installing project dependencies...
    py -3 -m pip install -r requirements.txt
    if errorlevel 1 goto install_failed
)
py -3 -m streamlit run app.py
goto end

:install_failed
echo.
echo Dependency installation failed. Review the messages above.
pause
exit /b 1

:end
pause

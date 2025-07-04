@echo off
echo ===================================================
echo Installing Iris Flower Detection App Dependencies
echo ===================================================
echo.

echo Checking Python installation...
python --version
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in your PATH.
    echo Please install Python from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo.
echo Installing required packages...
echo.
pip install -r requirements.txt

if %ERRORLEVEL% neq 0 (
    echo.
    echo Error installing packages. Please check your internet connection
    echo or try running the command manually: pip install -r requirements.txt
    pause
    exit /b 1
)

echo.
echo ===================================================
echo Installation complete!
echo.
echo You can now run the application by:
echo 1. Double-clicking on run_app.bat
echo 2. Or running: python app.py
echo ===================================================
echo.
pause

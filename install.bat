@echo off
echo Installing NeuroBeton dependencies...

:: Check and setup virtual environment
echo Checking virtual environment...
if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo Failed to create virtual environment!
        pause
        exit /b 1
    )
    echo Virtual environment created.
) else (
    echo Virtual environment found.
)

:: Install backend dependencies
echo Installing backend dependencies...
call .venv\Scripts\activate.bat
if not exist "backend\requirements.txt" (
    echo requirements.txt not found!
    pause
    exit /b 1
)
pip install -r backend\requirements.txt
if errorlevel 1 (
    echo Failed to install backend dependencies!
    pause
    exit /b 1
)
echo Backend dependencies installed.
call .venv\Scripts\deactivate.bat

:: Install frontend dependencies
echo Installing frontend dependencies...
cd frontend
if not exist "package.json" (
    echo package.json not found!
    cd ..
    pause
    exit /b 1
)
npm install
if errorlevel 1 (
    echo Failed to install frontend dependencies!
    cd ..
    pause
    exit /b 1
)
echo Frontend dependencies installed.
cd ..

echo.
echo Installation complete!
echo Run start.bat to launch the application.
pause
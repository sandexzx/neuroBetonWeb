@echo off
echo Starting NeuroBeton...

:: Check if virtual environment exists
if not exist ".venv" (
    echo Virtual environment not found!
    echo Run install.bat first.
    pause
    exit /b 1
)

:: Check if frontend dependencies exist
if not exist "frontend\node_modules" (
    echo Frontend dependencies not found!
    echo Run install.bat first.
    pause
    exit /b 1
)

echo Starting backend...
start "NeuroBeton Backend" cmd /k "cd backend && ..\.venv\Scripts\activate && uvicorn main:app --reload"

echo Waiting 3 seconds...
timeout /t 3 >nul

echo Starting frontend...
start "NeuroBeton Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo Both servers started!
echo Backend: http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Press any key to exit (servers will continue running)
pause
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

:: Распределение моделей из архива
echo Distributing models from archive...
if not exist "modelArchive" (
    echo Model archive not found!
    pause
    exit /b 1
)

:: Создаем базовую директорию для моделей если её нет
if not exist "backend\models" mkdir "backend\models"

:: Создаем необходимые директории для каждой модели
if not exist "backend\models\strength" mkdir "backend\models\strength"
if not exist "backend\models\classification" mkdir "backend\models\classification"
if not exist "backend\models\cracks" mkdir "backend\models\cracks"

:: Проверяем наличие каждой модели в архиве и копируем их
if exist "modelArchive\best_strength_prediction_model.pt" (
    copy "modelArchive\best_strength_prediction_model.pt" "backend\models\strength\" /Y
    echo Copied strength prediction model
) else (
    echo Warning: strength prediction model not found in archive
)

if exist "modelArchive\best_concrete_type_classification_model.pt" (
    copy "modelArchive\best_concrete_type_classification_model.pt" "backend\models\classification\" /Y
    echo Copied concrete type classification model
) else (
    echo Warning: concrete type classification model not found in archive
)

if exist "modelArchive\label_mapping.pkl" (
    copy "modelArchive\label_mapping.pkl" "backend\models\classification\" /Y
    echo Copied classification label mapping
) else (
    echo Warning: classification label mapping not found in archive
)

if exist "modelArchive\best_cracks_detection_model.pt" (
    copy "modelArchive\best_cracks_detection_model.pt" "backend\models\cracks\" /Y
    echo Copied cracks detection model
) else (
    echo Warning: cracks detection model not found in archive
)

:: Удаляем архив после копирования
rmdir /S /Q "modelArchive"
echo Models distributed successfully.

echo.
echo Installation complete!
echo Run start.bat to launch the application.
pause
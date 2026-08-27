@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ======================================================================
echo           Dog Emotion Image Processing - Automated Pipeline
echo ======================================================================
echo.

:: 1. ติดตั้ง Dependencies จาก requirements.txt
echo [Step 0/5] Installing Python Dependencies...
pip install -r requirements.txt
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to install dependencies!
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: 2. รัน Data Collection
echo [Step 1/5] Running Data Collection (src/data_collection.py)...
python src\data_collection.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Error in Data Collection!
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: 3. รัน Exploratory Data Analysis (EDA)
echo [Step 2/5] Running EDA (src/eda.py)...
python src\eda.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Error in EDA!
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: 4. รัน Preprocessing
echo [Step 3/5] Running Preprocessing (src/preprocessing.py)...
python src\preprocessing.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Error in Preprocessing!
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: 5. รัน Image Processing
echo [Step 4/5] Running Image Processing (src/image_processing.py)...
python src\image_processing.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Error in Image Processing!
    pause
    exit /b %ERRORLEVEL%
)
echo.

:: 6. รัน Data Splitting
echo [Step 5/5] Running Data Splitting (src/data_splitting.py)...
python src\data_splitting.py
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Error in Data Splitting!
    pause
    exit /b %ERRORLEVEL%
)
echo.

echo ======================================================================
echo           Pipeline Completed Successfully! All tasks finished.
echo ======================================================================
echo.
pause

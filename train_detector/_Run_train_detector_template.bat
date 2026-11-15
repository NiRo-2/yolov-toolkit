@echo off
REM Trains a YOLO26 detector from a dataset DIRECTORY with hardware-aware hyperparameters.
:: Template for training detector.
:: Create a personal copy (e.g. _Run_train_detector_personal.bat) and fill in your own paths.
::
:: --input is a dataset folder: YOLO (data.yaml present or synthesized), COCO JSON, or VOC XML.
:: Format and task (detect/segment/pose) are auto-detected; COCO/VOC are converted once
:: into a reusable <folder>_yolo copy next to the dataset.
:: Auto-config selects model, imgsz, batch, and workers from hardware + dataset.
:: Fresh runs use nbs=64 gradient accumulation (effective batch 64 when batch < 64).
:: Optional overrides: append --task segment, --converted-dir C:\out, --yes, --model,
:: --batch, --imgsz, --workers, etc. to the python line.
::
:: Required: dataset folder path and project name.
:: Example personal file (gitignored):
::   set datasetPath=c:\indir\my_dataset
::   set name=my_detector

setlocal

set "scriptDir=%~dp0"
set "scriptPath=%scriptDir%train_detector.py"

if not exist "%scriptPath%" (
    echo ERROR: Script not found:
    echo "%scriptPath%"
    pause
    exit /b 1
)

set datasetPath=_YOUR_DATASET_FOLDER_HERE_
set name=_YOUR_PROJECT_NAME_HERE_

if "%datasetPath%"=="" (
    echo ERROR: datasetPath cannot be empty.
    pause
    exit /b 1
)
echo %datasetPath% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set datasetPath in this batch file before running.
    pause
    exit /b 1
)
if "%name%"=="" (
    echo ERROR: name cannot be empty.
    pause
    exit /b 1
)
echo %name% | findstr /I "_YOUR_" >nul && (
    echo ERROR: Set name in this batch file before running.
    pause
    exit /b 1
)
if not exist "%datasetPath%\" (
    echo ERROR: Dataset folder not found:
    echo "%datasetPath%"
    pause
    exit /b 1
)

python "%scriptPath%" --input "%datasetPath%" --name %name%
set "exitCode=%ERRORLEVEL%"
if not "%exitCode%"=="0" (
    pause
    exit /b %exitCode%
)
pause
exit /b 0

@echo off
REM Exit immediately if a command exits with a non-zero status
setlocal enabledelayedexpansion

REM Step 1: Take a data sample
echo Taking a data sample...
python src\data.py project_stage=1
if not errorlevel 1 (
    echo Data sample taken successfully.
) else (
    echo Failed to take data sample.
    exit /b 1
)

REM Step 2: Validate the data sample
echo Validating the data sample...
python tests\data_expectations.py
if not errorlevel 1 (
    echo Data validation successful.
) else (
    echo Data validation failed. Not versioning the data sample.
    exit /b 1
)

REM Step 3: Version the data sample with DVC
echo Versioning the data sample with DVC...

dvc add "data/samples"

REM End of script
echo Script execution completed.
exit /b 0

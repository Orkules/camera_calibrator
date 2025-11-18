@echo off
setlocal

if not exist venv (
    python -m venv venv
)

call venv\Scripts\activate.bat

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller

pyinstaller --noconfirm --clean ^
  --name "CameraCalibrator" ^
  --add-data "templates;templates" ^
  --add-data "config.yaml;." ^
  --add-data "calibration_files;calibration_files" ^
  --add-data "luminar_eth_operation.txt;." ^
  app.py

echo.
echo PyInstaller build complete. Check the dist\CameraCalibrator folder.
endlocal


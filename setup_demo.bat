@echo off
echo ================================
echo Dark Pattern Hunter - Setup
echo ================================
echo.

echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Installing Playwright browsers...
playwright install chromium

echo.
echo ================================
echo Setup complete!
echo.
echo To run the demo:
echo   python demo.py
echo.
echo To run a specific test:
echo   python demo.py false_urgency
echo   python demo.py roach_motel
echo   python demo.py clean_stock
echo ================================

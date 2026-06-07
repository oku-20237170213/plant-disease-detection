@echo off
echo ================================================
echo   Bitki Hastaligi Teshis Sistemi - Web Uygulamasi
echo ================================================
echo.

REM Check if virtual environment exists
IF NOT EXIST "venv\" (
    echo [1/3] Sanal ortam olusturuluyor...
    python -m venv venv
)

echo [2/3] Bagimlilıklar yukleniyor...
call venv\Scripts\activate.bat
pip install -r requirements.txt -q

echo [3/3] Sunucu baslatiliyor...
echo.
echo  --> Tarayicinizda acin: http://127.0.0.1:5000
echo  --> Durdurmak icin: Ctrl+C
echo.
python app.py
pause

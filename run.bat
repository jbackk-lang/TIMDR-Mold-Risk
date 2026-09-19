@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo  TIMDR-Mold-Risk -- instalacja zaleznosci i uruchomienie
echo ============================================================

where python >nul 2>nul
if errorlevel 1 (
    echo BLAD: nie znaleziono "python" w PATH. Zainstaluj Pythona 3
    echo ^(https://www.python.org/downloads/^) i zaznacz "Add to PATH"
    echo podczas instalacji.
    pause
    exit /b 1
)

echo.
echo Instaluje zaleznosci z requirements.txt ^(w tym bleak dla BLE^)...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo BLAD: instalacja zaleznosci nie powiodla sie - zobacz komunikaty wyzej.
    pause
    exit /b 1
)

echo.
echo Uruchamiam serwer TIMDR-Mold-Risk w osobnym oknie...
start "TIMDR-Mold-Risk API" cmd /k python api.py

echo Czekam, az serwer wstanie...
timeout /t 3 /nobreak >nul

echo Otwieram dashboard w przegladarce...
start "" http://127.0.0.1:5002/dashboard

echo.
echo Gotowe. Serwer dziala w osobnym oknie konsoli - zostaw je otwarte.
echo Zeby zatrzymac serwer, zamknij to okno konsoli albo nacisnij Ctrl+C w nim.
echo.
echo Zakladka "Czujnik na zywo (BLE)" w dashboardzie wymaga sensora
echo Xiaomi Mijia LYWSD03MMC z wgranym custom firmware (pvvx/ATC_MiThermometer)
echo - patrz README.md, sekcja "Czujnik Bluetooth".
pause

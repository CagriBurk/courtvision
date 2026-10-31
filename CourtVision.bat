@echo off
rem CourtVision'u acar (API'yi Electron kendisi .venv ile baslatir).
rem Bu pencereyi kapatmayin: API ve tahmin loglari burada gorunur.
cd /d "%~dp0electron-app"
call npm run dev
pause

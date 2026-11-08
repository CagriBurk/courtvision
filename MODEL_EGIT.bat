@echo off
rem CourtVision modellerini yeniden egitir ve degerlendirir (yaklasik 2 saat).
rem Once uygulamayi kapatin. Bitince uygulamayi yeniden acin.
rem Adimlar: ozellik tablosu -> kazanan + fark + toplam egitimi (100 deneme) -> degerlendirme
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

echo.
echo [1/3] Ozellik tablosu hazirlaniyor...
%PY% src\02_build_features.py || goto :hata

echo.
echo [2/3] Egitim: kazanan, fark, toplam (100 deneme, ~75 dk)...
%PY% src\03_train_model.py --task all --trials 100 || goto :hata

echo.
echo [3/3] Degerlendirme (holdout, kalibrasyon, hata dagilimi)...
%PY% src\04_evaluate_model.py || goto :hata

echo.
echo Tamamlandi. Uygulamayi yeniden acabilirsiniz.
pause
exit /b 0

:hata
echo.
echo HATA: yukaridaki mesaji kaydedin. Hatadan onceki adimlar tamamlandi;
echo egitim yarida kaldiysa bazi modeller yenilenmis olabilir. Tekrar calistirin.
pause
exit /b 1

batch
@echo off
title Analise IC - Executando
color 0A

if not exist ".venv\Scripts\activate.bat" (
    echo [ERRO] Execute primeiro: instalar.bat
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

:: Procura porta livre a partir de 8501
setlocal enabledelayedexpansion
set PORT=8501
:check_port
netstat -ano | findstr :!PORT! | findstr LISTENING >nul
if not errorlevel 1 (
    set /a PORT=!PORT!+1
    if !PORT! gtr 8600 (
        echo Nenhuma porta livre encontrada entre 8501 e 8600.
        pause
        exit /b 1
    )
    goto check_port
)

echo ============================================================
echo   INICIANDO APLICACAO - ANALISE IC
echo   Porta: !PORT!
echo ============================================================
echo.
echo A aplicacao abrira em http://localhost:!PORT!
echo Para encerrar, pressione CTRL+C.
echo.

python -m streamlit run app.py --server.port !PORT! --browser.gatherUsageStats false
pause
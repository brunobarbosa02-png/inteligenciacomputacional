@echo off
title Instalador - Analise IC
color 0B

echo ============================================================
echo   INSTALADOR - ANALISE DE INTELIGENCIA COMPUTACIONAL
echo ============================================================
echo.

:: Verifica se o Python esta instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRO] Python nao encontrado.
    echo.
    echo Baixe o Python 3.10 ou superior em:
    echo   https://www.python.org/downloads/
    echo.
    echo IMPORTANTE: marque "Add Python to PATH" durante a instalacao.
    echo.
    pause
    exit /b 1
)

echo [OK] Python encontrado.
python --version
echo.

:: Cria ambiente virtual
if not exist ".venv" (
    echo [1/3] Criando ambiente virtual...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERRO] Falha ao criar ambiente virtual.
        pause
        exit /b 1
    )
    echo [OK] Ambiente virtual criado.
) else (
    echo [OK] Ambiente virtual ja existe.
)
echo.

:: Ativa ambiente virtual
echo [2/3] Ativando ambiente virtual...
call .venv\Scripts\activate.bat

:: Atualiza pip
python -m pip install --upgrade pip --quiet

:: Instala dependencias
echo [3/3] Instalando dependencias (pode demorar alguns minutos)...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERRO] Falha ao instalar dependencias.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   INSTALACAO CONCLUIDA COM SUCESSO!
echo ============================================================
echo.
echo Para executar a aplicacao, use o arquivo: executar.bat
echo.
pause
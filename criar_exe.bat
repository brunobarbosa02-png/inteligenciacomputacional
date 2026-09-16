@echo off
title Criar Executavel - Analise IC
color 0E

echo ============================================================
echo   GERADOR DE EXECUTAVEL - ANALISE IC
echo ============================================================
echo.
echo Este script gera um executavel .exe que roda sem Python instalado.
echo.
echo AVISO: o executavel gerado tera cerca de 300-500 MB.
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo [ERRO] Execute primeiro: instalar.bat
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat

echo [1/3] Limpando builds anteriores...
if exist "build" rmdir /S /Q "build"
if exist "dist" rmdir /S /Q "dist"

echo [2/3] Gerando executavel com PyInstaller...
pyinstaller --noconfirm --onedir --windowed ^
    --name "Analise_IC" ^
    --add-data "app.py;." ^
    --hidden-import=sklearn ^
    --hidden-import=sklearn.utils._cython_blas ^
    --hidden-import=sklearn.neighbors.typedefs ^
    --hidden-import=sklearn.neighbors.quad_tree ^
    --hidden-import=sklearn.tree._utils ^
    --hidden-import=scipy ^
    --hidden-import=scipy.stats ^
    --hidden-import=streamlit ^
    --hidden-import=streamlit.runtime.scriptrunner.magic_funcs ^
    --hidden-import=ucimlrepo ^
    --hidden-import=openpyxl ^
    --collect-all streamlit ^
    --collect-all sklearn ^
    --collect-all scipy ^
    --collect-all ucimlrepo ^
    lanzador.py

if errorlevel 1 (
    echo [ERRO] Falha ao gerar executavel.
    pause
    exit /b 1
)

echo [3/3] Copiando arquivos auxiliares...
copy app.py dist\Analise_IC\ >nul
copy requirements.txt dist\Analise_IC\ >nul

echo.
echo ============================================================
echo   EXECUTAVEL GERADO EM: dist\Analise_IC\
echo ============================================================
echo.
echo Execute: dist\Analise_IC\Analise_IC.exe
echo.
pause
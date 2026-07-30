@echo off
setlocal
cd /d "%~dp0"

set "TCC_PYTHON=.venv\Scripts\python.exe"

if not exist "%TCC_PYTHON%" (
    echo Ambiente Python nao encontrado em .venv.
    echo Crie o ambiente e instale requirements.txt antes de continuar.
    pause
    exit /b 1
)

"%TCC_PYTHON%" -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo Streamlit ainda nao esta instalado.
    echo Execute: .venv\Scripts\python.exe -m pip install -r requirements.txt
    pause
    exit /b 1
)

"%TCC_PYTHON%" -m streamlit run "Fonte\interface\app.py"
endlocal

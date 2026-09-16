"""
Wrapper para executar o Streamlit a partir de um .exe
Coloque este arquivo junto com app.py na raiz do projeto.
"""
import os, sys, subprocess
from pathlib import Path

def main():
    # Detecta diretório do script (funciona tanto em .py quanto em .exe)
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS) if hasattr(sys, '_MEIPASS') else Path(sys.executable).parent
    else:
        base = Path(__file__).parent

    app_path = base / "app.py"
    if not app_path.exists():
        app_path = base.parent / "app.py"

    # Executa streamlit
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path),
           "--server.port", "8501", "--browser.gatherUsageStats", "false"]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
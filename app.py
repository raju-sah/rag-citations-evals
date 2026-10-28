"""Hugging Face Spaces & production Streamlit entrypoint."""
import runpy
import sys
from pathlib import Path

# Ensure src/ is in sys.path
REPO_ROOT = Path(__file__).resolve().parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Launch app/streamlit_app.py
STREAMLIT_APP = REPO_ROOT / "app" / "streamlit_app.py"
runpy.run_path(str(STREAMLIT_APP), run_name="__main__")

"""DeepTrace // Digital Forensics Terminal UI Root Entry Point.

Ensures running `streamlit run app.py` from the root workspace directory executes
the full application seamlessly with correct path resolution.
"""

from pathlib import Path
import sys

_HACKATHON_DIR = Path(__file__).resolve().parent / "Hackathon"
if _HACKATHON_DIR.is_dir():
    if str(_HACKATHON_DIR) not in sys.path:
        sys.path.insert(0, str(_HACKATHON_DIR))

# Import and execute the terminal application
from Hackathon.app import render_app, logging

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    render_app()

"""
LegalitySimplified — Entry Point

Usage:
    python run.py          # Start the Streamlit app
"""
import subprocess
import sys


def main():
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/main.py",
         "--server.headless", "true"],
        check=True,
    )


if __name__ == "__main__":
    main()

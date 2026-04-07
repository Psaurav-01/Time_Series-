"""
Entry point for the GARCH / DCC Volatility Dashboard.

Local development:
    python main.py

Docker / Render (via Gunicorn — see Dockerfile):
    gunicorn --bind 0.0.0.0:${PORT:-8050} --timeout 300 --workers 1 main:server
"""

import os
import sys

# Ensure project root is on sys.path so `src.*` imports work everywhere
sys.path.insert(0, os.path.dirname(__file__))

from src.dashboard.app import app, server  # noqa: F401  (server exported for gunicorn)

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 8050))
    debug = os.environ.get("DASH_DEBUG", "false").lower() == "true"

    print(f"Starting GARCH Dashboard on http://0.0.0.0:{port}  (debug={debug})")
    app.run(host="0.0.0.0", port=port, debug=debug)

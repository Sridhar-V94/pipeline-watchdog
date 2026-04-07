"""
server/app.py — OpenEnv entry point (required by validator).
Imports and re-exports the FastAPI app from the root server.py.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app  # noqa: F401 — re-export for uvicorn server.app:app

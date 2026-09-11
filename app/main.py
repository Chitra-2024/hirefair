"""Top-level FastAPI application entry point for HireFair.

Enables starting the backend with either:
  uvicorn app.main:app --reload
or:
  uvicorn app.api.main:app --reload
"""

from app.api.main import app

__all__ = ["app"]

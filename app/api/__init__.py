"""FastAPI application package for HireFair."""

from app.api.main import app
from app.api.service import screen_documents

__all__ = ["app", "screen_documents"]

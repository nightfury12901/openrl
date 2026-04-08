"""
FastAPI Application for the Legal Case Assistant OpenEnv Environment.

Exposes HTTP endpoints as strictly defined by `openenv-core`.
"""

from __future__ import annotations

from openenv.core.env_server import create_fastapi_app
from legal_env.models import LegalAction, LegalObservation
from legal_env.server.legal_environment import LegalEnvironment

env = LegalEnvironment()

# Standard OpenEnv FastAPI server hook. Automatically manages /reset, /step, /state, and /health
app = create_fastapi_app(env, LegalAction, LegalObservation)

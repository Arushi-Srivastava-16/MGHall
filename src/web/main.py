"""
FastAPI Main Application.

Main entry point for the CHG Framework web dashboard backend.
"""

import sys
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    # Load .env from project root (3 levels up from this file)
    env_path = Path(__file__).parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    # python-dotenv not installed, skip .env loading
    pass

from src.web.api import (
    chains,
    models,
    fingerprints,
    consensus,
    patterns,
    inference,
    training,
    results,
)

# Create FastAPI app
app = FastAPI(
    title="CHG Framework Dashboard API",
    description="API for the Causal Hallucination Graph Framework Dashboard",
    version="1.0.0",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(chains.router, prefix="/api/chains", tags=["chains"])
app.include_router(models.router, prefix="/api/models", tags=["models"])
app.include_router(fingerprints.router, prefix="/api/fingerprints", tags=["fingerprints"])
app.include_router(consensus.router, prefix="/api/consensus", tags=["consensus"])
app.include_router(patterns.router, prefix="/api/patterns", tags=["patterns"])
app.include_router(inference.router, prefix="/api/inference", tags=["inference"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(results.router, prefix="/api/results", tags=["results"])


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "CHG Framework Dashboard API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "type": type(exc).__name__},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


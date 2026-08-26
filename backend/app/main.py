from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.api.routes import health
from app.api.routes import documents

# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    description=(
        "MemoryOS API — the backend for your personal knowledge layer. "
        "Upload documents and query your memory in plain language."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — allow the React dev server
# ---------------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Global exception handler
# ---------------------------------------------------------------------------


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": str(request.url),
        },
    )


# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(health.router, prefix="/api")
app.include_router(documents.router, prefix="/api/documents")

# ---------------------------------------------------------------------------
# Root redirect
# ---------------------------------------------------------------------------


@app.get("/", include_in_schema=False)
async def root() -> JSONResponse:
    return JSONResponse(
        content={
            "message": f"Welcome to {settings.APP_NAME}",
            "docs": "/docs",
            "health": "/api/health",
        }
    )

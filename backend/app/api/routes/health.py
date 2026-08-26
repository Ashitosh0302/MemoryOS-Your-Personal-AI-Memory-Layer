from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    service: str


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns the current health status of the MemoryOS API.",
    tags=["Health"],
)
async def health_check() -> HealthResponse:
    """Verify the API is running and reachable."""
    return HealthResponse(status="healthy", service="MemoryOS API")

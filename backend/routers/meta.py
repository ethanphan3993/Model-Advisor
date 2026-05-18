"""Meta endpoints — surface use cases / harnesses / source status to frontend."""

from fastapi import APIRouter

from backend.db import connect, get_source_runs
from backend.models.schemas import (
    HarnessInfo, SourcesResponse, SourceStatus, UseCaseInfo,
)
from backend.services.data_loader import harnesses, use_cases

router = APIRouter()


@router.get("/use-cases", response_model=list[UseCaseInfo])
async def list_use_cases():
    return [UseCaseInfo(id=u["id"], name=u["name"], tagline=u["tagline"], icon=u.get("icon", ""))
            for u in use_cases()]


@router.get("/harnesses", response_model=list[HarnessInfo])
async def list_harnesses():
    return [HarnessInfo(
        id=h["id"], name=h["name"], category=h["category"],
        description=h["description"], homepage=h.get("homepage", ""),
    ) for h in harnesses()]


@router.get("/sources", response_model=SourcesResponse)
async def list_sources():
    with connect() as conn:
        runs = get_source_runs(conn)
    return SourcesResponse(sources=[SourceStatus(**r) for r in runs])

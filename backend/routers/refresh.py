"""Trigger a refresh of one or all data sources."""

from fastapi import APIRouter, HTTPException

from backend.services.refresh import refresh_all, refresh_one

router = APIRouter()


@router.post("/refresh")
async def refresh_endpoint(source: str | None = None):
    if source:
        result = await refresh_one(source)
        return {"results": [result.__dict__]}
    results = await refresh_all()
    return {"results": [r.__dict__ for r in results]}

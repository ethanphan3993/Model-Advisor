"""Hardware scan endpoint."""

from fastapi import APIRouter, HTTPException

from backend.models.schemas import ScanResponse
from backend.services.hardware import scan_hardware

router = APIRouter()


@router.get("/scan", response_model=ScanResponse,
            responses={500: {"description": "Scan failed (non-macOS or error)"}})
async def scan():
    result = scan_hardware()
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])
    return result

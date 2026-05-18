"""Recommendation endpoints (use_case × harness × hardware)."""

from fastapi import APIRouter, HTTPException

from backend.models.schemas import (
    HardwareSnapshot, InstallOption, ProvenanceModel, RecommendRequest,
    RecommendResponse, RecommendationModel, ScoreEvidenceModel,
)
from backend.services.hardware import scan_hardware
from backend.services.recommender import HardwareSnapshot as HSDataclass, recommend

router = APIRouter()


def _hw_snapshot() -> HSDataclass:
    hw_data = scan_hardware()
    if "error" in hw_data:
        raise HTTPException(status_code=500, detail=hw_data["error"])
    chip = hw_data["hardware"]["chip"]
    storage = hw_data.get("storage") or []
    free_gb = max((s.get("free_gb", 0) for s in storage), default=0.0)
    return HSDataclass(
        chip=chip["chip"],
        generation=chip["generation"],
        gpu_cores=chip["gpu_cores"],
        total_memory_gb=hw_data["hardware"]["total_memory_gb"],
        available_memory_gb=hw_data["memory"]["available_gb"],
        neural_engine_cores=chip["neural_engine_cores"],
        storage_free_gb=free_gb,
    )


def _to_pydantic(rec) -> RecommendationModel:
    return RecommendationModel(
        rank=rec.rank,
        canonical_id=rec.canonical_id,
        display_name=rec.display_name,
        family=rec.family,
        parameter_size=rec.parameter_size,
        variant=rec.variant,
        is_moe=rec.is_moe,
        fit_score=rec.fit_score,
        use_case_score=rec.use_case_score,
        hardware_fit=rec.hardware_fit,
        harness_fit=rec.harness_fit,
        confidence=rec.confidence,
        quantization_recommended=rec.quantization_recommended,
        estimated_size_mb=rec.estimated_size_mb,
        estimated_kv_cache_mb=rec.estimated_kv_cache_mb,
        estimated_tokens_per_sec=rec.estimated_tokens_per_sec,
        install_options=[InstallOption(**i) for i in rec.install_options],
        warnings=rec.warnings,
        provenance=ProvenanceModel(
            use_case_components=[ScoreEvidenceModel(**e.__dict__) for e in rec.provenance.use_case_components],
            hardware_components=rec.provenance.hardware_components,
            harness_components=rec.provenance.harness_components,
            missing_data=rec.provenance.missing_data,
        ),
        why=rec.why,
    )


@router.post("/recommend", response_model=RecommendResponse)
async def post_recommend(req: RecommendRequest):
    hw = _hw_snapshot()
    recs = recommend(
        use_case_id=req.use_case,
        harness_id=req.harness,
        hw=hw,
        limit=req.limit,
        include_too_big=req.include_too_big,
    )
    return RecommendResponse(
        use_case=req.use_case,
        harness=req.harness,
        hardware_snapshot=HardwareSnapshot(**hw.__dict__),
        recommendations=[_to_pydantic(r) for r in recs],
        total_candidates=len(recs),
    )

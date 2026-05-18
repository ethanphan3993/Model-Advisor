"""Image-generation endpoints — separate from the text-LLM /api/recommend
because the surface (use cases, harnesses, cost model) is genuinely different.

See backend/services/images/recommender.py for the cost-model rationale.
"""

import time

from fastapi import APIRouter, HTTPException

from backend.models.image_schemas import (
    ImageHardwareSnapshotModel, ImageHarnessInfo, ImageInstallOption,
    ImageModelCard, ImageProvenanceModel, ImageRecommendRequest,
    ImageRecommendResponse, ImageRecommendationModel, ImageScoreEvidenceModel,
    ImageUseCaseInfo,
)
from backend.services.hardware import scan_hardware
from backend.services.images.catalog import (
    image_harnesses, image_models, image_use_cases,
)
from backend.services.images.recommender import (
    ImageHardwareSnapshot, recommend,
)

router = APIRouter()


# Same 15s hardware-cache pattern as /api/recommend.
_HW_CACHE_TTL_S = 15
_hw_cache: dict[str, tuple[float, ImageHardwareSnapshot]] = {}


def _hw_snapshot() -> ImageHardwareSnapshot:
    cached = _hw_cache.get("snapshot")
    if cached and time.time() - cached[0] < _HW_CACHE_TTL_S:
        return cached[1]

    hw_data = scan_hardware()
    if "error" in hw_data:
        raise HTTPException(status_code=500, detail=hw_data["error"])
    chip = hw_data["hardware"]["chip"]
    storage = hw_data.get("storage") or []
    free_gb = max((s.get("free_gb", 0) for s in storage), default=0.0)
    snapshot = ImageHardwareSnapshot(
        chip=chip["chip"],
        generation=chip["generation"],
        gpu_cores=chip["gpu_cores"],
        total_memory_gb=hw_data["hardware"]["total_memory_gb"],
        available_memory_gb=hw_data["memory"]["available_gb"],
        storage_free_gb=free_gb,
    )
    _hw_cache["snapshot"] = (time.time(), snapshot)
    return snapshot


@router.get("/images/use-cases", response_model=list[ImageUseCaseInfo])
async def list_image_use_cases():
    return [ImageUseCaseInfo(
        id=u["id"], name=u["name"], tagline=u["tagline"], icon=u.get("icon", ""),
    ) for u in image_use_cases()]


@router.get("/images/harnesses", response_model=list[ImageHarnessInfo])
async def list_image_harnesses():
    return [ImageHarnessInfo(
        id=h["id"], name=h["name"], category=h["category"],
        description=h["description"], homepage=h.get("homepage", ""),
    ) for h in image_harnesses()]


@router.get("/images/catalog", response_model=list[ImageModelCard])
async def get_image_catalog():
    return [ImageModelCard(
        canonical_id=m.canonical_id, family=m.family, variant=m.variant,
        display_name=m.display_name, architecture=m.architecture,
        total_params_b=m.total_params_b, default_steps=m.default_steps,
        vram_gb=m.vram_gb, license=m.license, supports=m.supports,
        harnesses_compatible=m.harnesses_compatible, hf_id=m.hf_id, notes=m.notes,
    ) for m in image_models()]


def _to_pydantic(rec) -> ImageRecommendationModel:
    return ImageRecommendationModel(
        rank=rec.rank,
        canonical_id=rec.canonical_id,
        display_name=rec.display_name,
        family=rec.family,
        variant=rec.variant,
        architecture=rec.architecture,
        fit_score=rec.fit_score,
        use_case_score=rec.use_case_score,
        hardware_fit=rec.hardware_fit,
        harness_fit=rec.harness_fit,
        confidence=rec.confidence,
        confidence_pct=rec.confidence_pct,
        benchmarks_measured=rec.benchmarks_measured,
        benchmarks_expected=rec.benchmarks_expected,
        quantization_recommended=rec.quantization_recommended,
        estimated_vram_gb=rec.estimated_vram_gb,
        estimated_time_per_image_s=rec.estimated_time_per_image_s,
        default_steps=rec.default_steps,
        fits_currently_free=rec.fits_currently_free,
        license=rec.license,
        supports=rec.supports,
        install_options=[ImageInstallOption(**i) for i in rec.install_options],
        warnings=rec.warnings,
        provenance=ImageProvenanceModel(
            use_case_components=[ImageScoreEvidenceModel(**e.__dict__) for e in rec.provenance.use_case_components],
            hardware_components=rec.provenance.hardware_components,
            harness_components=rec.provenance.harness_components,
            missing_data=rec.provenance.missing_data,
        ),
        why=rec.why,
        notes=rec.notes,
    )


@router.post("/images/recommend", response_model=ImageRecommendResponse)
async def post_image_recommend(req: ImageRecommendRequest):
    hw = _hw_snapshot()
    recs = recommend(
        use_case_id=req.use_case,
        harness_id=req.harness,
        hw=hw,
        limit=req.limit,
        include_too_big=req.include_too_big,
    )
    return ImageRecommendResponse(
        use_case=req.use_case,
        harness=req.harness,
        hardware_snapshot=ImageHardwareSnapshotModel(
            chip=hw.chip, generation=hw.generation, gpu_cores=hw.gpu_cores,
            total_memory_gb=hw.total_memory_gb,
            available_memory_gb=hw.available_memory_gb,
            storage_free_gb=hw.storage_free_gb,
            fp16_tflops=hw.fp16_tflops(),
        ),
        recommendations=[_to_pydantic(r) for r in recs],
        total_candidates=len(recs),
    )

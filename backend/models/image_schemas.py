"""Pydantic schemas for the image-generation track."""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class ImageUseCaseInfo(BaseModel):
    id: str
    name: str
    tagline: str
    icon: str = ""


class ImageHarnessInfo(BaseModel):
    id: str
    name: str
    category: str
    description: str
    homepage: str = ""


class ImageScoreEvidenceModel(BaseModel):
    benchmark: str
    value: float
    normalized: float
    source: str
    confidence: str


class ImageProvenanceModel(BaseModel):
    use_case_components: list[ImageScoreEvidenceModel] = []
    hardware_components: dict[str, float] = {}
    harness_components: dict[str, float] = {}
    missing_data: list[str] = []


class ImageInstallOption(BaseModel):
    harness: str
    harness_id: str
    command: str
    homepage: str = ""
    download_url: str = ""


class ImageRecommendationModel(BaseModel):
    rank: int
    canonical_id: str
    display_name: str
    family: str
    variant: str
    architecture: str
    fit_score: float
    use_case_score: float
    hardware_fit: float
    harness_fit: float
    confidence: str
    confidence_pct: int = 0
    benchmarks_measured: int = 0
    benchmarks_expected: int = 0
    quantization_recommended: str
    estimated_vram_gb: float
    estimated_time_per_image_s: float
    default_steps: int
    fits_currently_free: bool = True
    license: str = ""
    supports: list[str] = []
    install_options: list[ImageInstallOption] = []
    warnings: list[str] = []
    provenance: ImageProvenanceModel
    why: str
    notes: str = ""


class ImageRecommendRequest(BaseModel):
    use_case: str
    harness: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)
    include_too_big: bool = False


class ImageHardwareSnapshotModel(BaseModel):
    chip: str
    generation: str
    gpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    storage_free_gb: float
    fp16_tflops: float


class ImageRecommendResponse(BaseModel):
    use_case: str
    harness: Optional[str] = None
    hardware_snapshot: ImageHardwareSnapshotModel
    recommendations: list[ImageRecommendationModel]
    total_candidates: int


class ImageModelCard(BaseModel):
    canonical_id: str
    family: str
    variant: str
    display_name: str
    architecture: str
    total_params_b: float
    default_steps: int
    vram_gb: dict[str, float]
    license: str
    supports: list[str]
    harnesses_compatible: list[str]
    hf_id: str = ""
    notes: str = ""

from __future__ import annotations

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Hardware scan
# ---------------------------------------------------------------------------

class ChipInfo(BaseModel):
    chip: str
    generation: str
    neural_engine_cores: int
    cpu_cores: dict[str, int]
    gpu_cores: int = 0


class HardwareInfo(BaseModel):
    model: str
    model_identifier: str
    model_number: str
    serial_number: str
    chip: ChipInfo
    total_memory_gb: int
    memory_type: str
    boot_rom_version: str
    activation_lock: str


class GPUInfo(BaseModel):
    gpu_cores: int
    vendor: str
    metal_support: str
    model: str
    connection_type: str


class MemoryInfo(BaseModel):
    total_gb: float
    available_gb: float
    used_gb: float
    wired_down_gb: float
    compressed_gb: float
    memory_type: str
    manufacturer: str
    page_size_bytes: int


class StorageInfo(BaseModel):
    name: str
    capacity_gb: float
    free_gb: float
    free_pct: float
    media_type: str
    protocol: str
    smart_status: str


class DisplayInfo(BaseModel):
    model: str
    resolution: str
    refresh_rate: str
    type: str
    main: bool
    internal: bool
    ambient_brightness: bool


class OSInfo(BaseModel):
    name: str
    version: str
    build: str


class CPUExtra(BaseModel):
    brand_string: str
    logical_cores: int
    physical_cores: int


class PowerInfo(BaseModel):
    battery_health: str
    max_capacity: str
    cycle_count: int
    is_charging: bool
    charge_pct: int
    battery_model: str
    firmware_version: str


class AICapability(BaseModel):
    gpu_score: int
    memory_score: int
    neural_engine_score: int
    composite_score: float
    max_composite: int
    interpretation: str


class ScanResponse(BaseModel):
    hardware: HardwareInfo
    gpu: GPUInfo
    memory: MemoryInfo
    storage: list[StorageInfo]
    display: DisplayInfo
    os: OSInfo
    cpu_extra: CPUExtra
    power: Optional[PowerInfo] = None
    ai_capability: AICapability


# ---------------------------------------------------------------------------
# Models browse
# ---------------------------------------------------------------------------

class ArtifactInfo(BaseModel):
    source: str
    source_id: str
    quantization: str = ""
    size_mb: float = 0
    download_url: str = ""
    install_command: str = ""


class ModelSummary(BaseModel):
    canonical_id: str
    family: str
    parameter_size: str
    variant: str
    display_name: str
    total_params_b: float = 0
    active_params_b: float = 0
    is_moe: bool = False
    context_length: int = 0
    tool_calling: bool = False
    vision: bool = False
    license: str = ""
    artifacts: list[ArtifactInfo] = []
    benchmark_count: int = 0


class FacetCount(BaseModel):
    value: str
    count: int


class ModelFacets(BaseModel):
    families: list[FacetCount] = []
    sources: list[FacetCount] = []
    total_with_benchmarks: int = 0
    total_moe: int = 0
    total_tool_calling: int = 0
    total_vision: int = 0


class ModelListResponse(BaseModel):
    models: list[ModelSummary]
    total: int
    limit: int
    offset: int
    facets: ModelFacets = ModelFacets()


class BenchmarkScore(BaseModel):
    benchmark: str
    value: float
    max_value: float
    normalized: float
    source: str
    confidence: str


class ModelDetailResponse(ModelSummary):
    scores: list[BenchmarkScore] = []
    description: str = ""


# ---------------------------------------------------------------------------
# Recommend
# ---------------------------------------------------------------------------

class HardwareSnapshot(BaseModel):
    chip: str
    generation: str
    gpu_cores: int
    total_memory_gb: float
    available_memory_gb: float
    neural_engine_cores: int
    storage_free_gb: float


class ScoreEvidenceModel(BaseModel):
    benchmark: str
    value: float
    normalized: float
    source: str
    confidence: str


class ProvenanceModel(BaseModel):
    use_case_components: list[ScoreEvidenceModel] = []
    hardware_components: dict[str, float] = {}
    harness_components: dict[str, float] = {}
    missing_data: list[str] = []


class InstallOption(BaseModel):
    harness: Optional[str] = None
    source: str
    source_id: str
    command: str = ""
    url: str = ""
    size_mb: float = 0
    quantization: str = ""


class RecommendationModel(BaseModel):
    rank: int
    canonical_id: str
    display_name: str
    family: str
    parameter_size: str
    variant: str
    is_moe: bool = False
    fit_score: float
    use_case_score: float
    hardware_fit: float
    harness_fit: float
    confidence: str
    confidence_pct: int = 0
    benchmarks_measured: int = 0
    benchmarks_expected: int = 0
    quant_quality_factor: float = 1.0
    quantization_recommended: str
    estimated_size_mb: float
    estimated_kv_cache_mb: float = 0
    estimated_tokens_per_sec: tuple[int, int]
    bandwidth_gb_s: float = 0
    active_params_b: float = 0
    total_params_b: float = 0
    fits_currently_free: bool = True
    install_options: list[InstallOption]
    warnings: list[str]
    provenance: ProvenanceModel
    why: str


class RecommendRequest(BaseModel):
    use_case: str
    harness: Optional[str] = None
    limit: int = Field(default=10, ge=1, le=50)
    include_too_big: bool = False
    include_unscored: bool = False


class RecommendResponse(BaseModel):
    use_case: str
    harness: Optional[str] = None
    hardware_snapshot: HardwareSnapshot
    recommendations: list[RecommendationModel]
    total_candidates: int


# ---------------------------------------------------------------------------
# Meta endpoints
# ---------------------------------------------------------------------------

class UseCaseInfo(BaseModel):
    id: str
    name: str
    tagline: str
    icon: str = ""


class HarnessInfo(BaseModel):
    id: str
    name: str
    category: str
    description: str
    homepage: str = ""


class SourceStatus(BaseModel):
    source: str
    last_run_at: int
    last_status: str
    error_message: str = ""
    rows_written: int = 0
    duration_ms: int = 0


class SourcesResponse(BaseModel):
    sources: list[SourceStatus]


class HealthResponse(BaseModel):
    status: str
    app: str
    version: str

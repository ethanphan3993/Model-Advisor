// ---------------------------------------------------------------------------
// Hardware
// ---------------------------------------------------------------------------

export interface ChipInfo {
  chip: string
  generation: string
  neural_engine_cores: number
  cpu_cores: Record<string, number>
  gpu_cores: number
}

export interface HardwareInfo {
  model: string
  model_identifier: string
  model_number: string
  serial_number: string
  chip: ChipInfo
  total_memory_gb: number
  memory_type: string
  boot_rom_version: string
  activation_lock: string
}

export interface GPUInfo {
  gpu_cores: number
  vendor: string
  metal_support: string
  model: string
  connection_type: string
}

export interface MemoryInfo {
  total_gb: number
  available_gb: number
  used_gb: number
  wired_down_gb: number
  compressed_gb: number
  memory_type: string
  manufacturer: string
  page_size_bytes: number
}

export interface StorageInfo {
  name: string
  capacity_gb: number
  free_gb: number
  free_pct: number
  media_type: string
  protocol: string
  smart_status: string
}

export interface DisplayInfo {
  model: string
  resolution: string
  refresh_rate: string
  type: string
  main: boolean
  internal: boolean
  ambient_brightness: boolean
}

export interface OSInfo { name: string; version: string; build: string }
export interface CPUExtra { brand_string: string; logical_cores: number; physical_cores: number }
export interface PowerInfo {
  battery_health: string
  max_capacity: string
  cycle_count: number
  is_charging: boolean
  charge_pct: number
  battery_model: string
  firmware_version: string
}
export interface AICapability {
  gpu_score: number
  memory_score: number
  neural_engine_score: number
  composite_score: number
  max_composite: number
  interpretation: string
}

export interface ScanResponse {
  hardware: HardwareInfo
  gpu: GPUInfo
  memory: MemoryInfo
  storage: StorageInfo[]
  display: DisplayInfo
  os: OSInfo
  cpu_extra: CPUExtra
  power: PowerInfo | null
  ai_capability: AICapability
}

// ---------------------------------------------------------------------------
// Models
// ---------------------------------------------------------------------------

export interface ArtifactInfo {
  source: string
  source_id: string
  quantization: string
  size_mb: number
  download_url: string
  install_command: string
}

export interface ModelSummary {
  canonical_id: string
  family: string
  parameter_size: string
  variant: string
  display_name: string
  total_params_b: number
  active_params_b: number
  is_moe: boolean
  context_length: number
  tool_calling: boolean
  vision: boolean
  license: string
  artifacts: ArtifactInfo[]
  benchmark_count: number
}

export interface BenchmarkScore {
  benchmark: string
  value: number
  max_value: number
  normalized: number
  source: string
  confidence: string
}

export interface ModelDetail extends ModelSummary {
  scores: BenchmarkScore[]
  description: string
}

export interface FacetCount {
  value: string
  count: number
}

export interface ModelFacets {
  families: FacetCount[]
  sources: FacetCount[]
  total_with_benchmarks: number
  total_moe: number
  total_tool_calling: number
  total_vision: number
}

export interface ModelListResponse {
  models: ModelSummary[]
  total: number
  limit: number
  offset: number
  facets: ModelFacets
}

// ---------------------------------------------------------------------------
// Recommend
// ---------------------------------------------------------------------------

export interface ScoreEvidence {
  benchmark: string
  value: number
  normalized: number
  source: string
  confidence: string
}

export interface Provenance {
  use_case_components: ScoreEvidence[]
  hardware_components: Record<string, number>
  harness_components: Record<string, number>
  missing_data: string[]
}

export interface InstallOption {
  harness: string | null
  source: string
  source_id: string
  command: string
  url: string
  size_mb: number
  quantization: string
}

export interface Recommendation {
  rank: number
  canonical_id: string
  display_name: string
  family: string
  parameter_size: string
  variant: string
  is_moe: boolean
  fit_score: number
  use_case_score: number
  hardware_fit: number
  harness_fit: number
  confidence: 'high' | 'medium' | 'low'
  confidence_pct: number
  benchmarks_measured: number
  benchmarks_expected: number
  quant_quality_factor: number
  quantization_recommended: string
  estimated_size_mb: number
  estimated_kv_cache_mb: number
  estimated_tokens_per_sec: [number, number]
  bandwidth_gb_s: number
  active_params_b: number
  total_params_b: number
  fits_currently_free: boolean
  install_options: InstallOption[]
  warnings: string[]
  provenance: Provenance
  why: string
}

export interface HardwareSnapshot {
  chip: string
  generation: string
  gpu_cores: number
  total_memory_gb: number
  available_memory_gb: number
  neural_engine_cores: number
  storage_free_gb: number
}

export interface RecommendRequest {
  use_case: string
  harness?: string | null
  limit?: number
  include_too_big?: boolean
  include_unscored?: boolean
}

export interface RecommendResponse {
  use_case: string
  harness: string | null
  hardware_snapshot: HardwareSnapshot
  recommendations: Recommendation[]
  total_candidates: number
}

// ---------------------------------------------------------------------------
// Meta
// ---------------------------------------------------------------------------

export interface UseCase { id: string; name: string; tagline: string; icon: string }
export interface Harness { id: string; name: string; category: string; description: string; homepage: string }
export interface SourceStatus {
  source: string
  last_run_at: number
  last_status: string
  error_message: string
  rows_written: number
  duration_ms: number
}
export interface SourcesResponse { sources: SourceStatus[] }
export interface HealthResponse { status: string; app: string; version: string }

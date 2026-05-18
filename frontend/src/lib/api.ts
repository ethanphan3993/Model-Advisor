import type {
  HealthResponse, ScanResponse,
  ModelListResponse, ModelDetail,
  RecommendRequest, RecommendResponse,
  UseCase, Harness, SourcesResponse,
  ImageUseCase, ImageHarness, ImageRecommendRequest, ImageRecommendResponse,
  ImageModelCard,
} from '../types'

const API_BASE = '/api'

async function fetchJSON<T>(url: string, init?: RequestInit): Promise<T> {
  const resp = await fetch(url, init)
  if (!resp.ok) {
    const body = await resp.text().catch(() => '')
    throw new Error(`API error ${resp.status}: ${body || resp.statusText}`)
  }
  return resp.json() as Promise<T>
}

export const getHealth = () => fetchJSON<HealthResponse>(`${API_BASE}/health`)

export const getScan = () => fetchJSON<ScanResponse>(`${API_BASE}/scan`)

export interface ListModelsParams {
  q?: string
  family?: string
  vision?: boolean
  tool_calling?: boolean
  is_moe?: boolean
  has_benchmarks?: boolean
  min_params?: number
  max_params?: number
  source?: string
  sort?: 'popular' | 'name' | 'params_asc' | 'params_desc' | 'benchmarks' | 'family'
  limit?: number
  offset?: number
}

export const listModels = (params: ListModelsParams = {}) => {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === '') continue
    qs.set(k, String(v))
  }
  const q = qs.toString()
  return fetchJSON<ModelListResponse>(`${API_BASE}/models${q ? '?' + q : ''}`)
}

export const getModel = (canonicalId: string) =>
  fetchJSON<ModelDetail>(`${API_BASE}/models/${encodeURIComponent(canonicalId)}`)

export const recommend = (req: RecommendRequest) =>
  fetchJSON<RecommendResponse>(`${API_BASE}/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })

export const listUseCases = () => fetchJSON<UseCase[]>(`${API_BASE}/use-cases`)
export const listHarnesses = () => fetchJSON<Harness[]>(`${API_BASE}/harnesses`)
export const getSources = () => fetchJSON<SourcesResponse>(`${API_BASE}/sources`)

export const triggerRefresh = (source?: string) =>
  fetchJSON<{ results: any[] }>(`${API_BASE}/refresh${source ? '?source=' + source : ''}`, { method: 'POST' })

// Image-generation endpoints
export const listImageUseCases = () => fetchJSON<ImageUseCase[]>(`${API_BASE}/images/use-cases`)
export const listImageHarnesses = () => fetchJSON<ImageHarness[]>(`${API_BASE}/images/harnesses`)
export const getImageCatalog = () => fetchJSON<ImageModelCard[]>(`${API_BASE}/images/catalog`)
export const recommendImages = (req: ImageRecommendRequest) =>
  fetchJSON<ImageRecommendResponse>(`${API_BASE}/images/recommend`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  })

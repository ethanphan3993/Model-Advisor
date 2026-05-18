import { useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { Search, Loader2, X, ChevronDown, ChevronRight } from 'lucide-react'
import { listModels, type ListModelsParams } from '../lib/api'
import { cn, formatSize } from '../lib/utils'
import type { ModelFacets, ModelSummary } from '../types'

type SortKey = NonNullable<ListModelsParams['sort']>

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'popular', label: 'Most popular' },
  { value: 'benchmarks', label: 'Most benchmarked' },
  { value: 'params_desc', label: 'Largest first' },
  { value: 'params_asc', label: 'Smallest first' },
  { value: 'name', label: 'Name (A–Z)' },
  { value: 'family', label: 'Family' },
]

const SIZE_BUCKETS: { id: string; label: string; min?: number; max?: number }[] = [
  { id: 'any', label: 'Any size' },
  { id: 'tiny', label: '< 4B', max: 4 },
  { id: 'small', label: '4B – 14B', min: 4, max: 14 },
  { id: 'mid', label: '14B – 32B', min: 14, max: 32 },
  { id: 'large', label: '32B – 100B', min: 32, max: 100 },
  { id: 'huge', label: '100B+', min: 100 },
]

const SOURCE_OPTIONS: { id: string; label: string }[] = [
  { id: 'ollama', label: 'Ollama' },
  { id: 'lmstudio-community', label: 'LM Studio' },
  { id: 'huggingface_gguf', label: 'HuggingFace GGUF' },
]

const PAGE_SIZE = 50

const emptyFacets: ModelFacets = {
  families: [], sources: [],
  total_with_benchmarks: 0, total_moe: 0, total_tool_calling: 0, total_vision: 0,
}

interface FilterState {
  q: string
  size: string
  family: string | null
  source: string | null
  has_benchmarks: boolean
  tool_calling: boolean
  is_moe: boolean
  vision: boolean
  sort: SortKey
}

const defaultFilters: FilterState = {
  q: '', size: 'any', family: null, source: null,
  has_benchmarks: false, tool_calling: false, is_moe: false, vision: false,
  sort: 'popular',
}


export default function Browse() {
  const [filters, setFilters] = useState<FilterState>(defaultFilters)
  const [debouncedQ, setDebouncedQ] = useState('')
  const [models, setModels] = useState<ModelSummary[]>([])
  const [total, setTotal] = useState(0)
  const [facets, setFacets] = useState<ModelFacets>(emptyFacets)
  const [loading, setLoading] = useState(true)
  const [loadingMore, setLoadingMore] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const requestSeq = useRef(0)

  // Debounce search
  useEffect(() => {
    const t = setTimeout(() => setDebouncedQ(filters.q), 250)
    return () => clearTimeout(t)
  }, [filters.q])

  const queryParams = useMemo<ListModelsParams>(() => {
    const sb = SIZE_BUCKETS.find(s => s.id === filters.size)
    return {
      q: debouncedQ || undefined,
      family: filters.family || undefined,
      source: filters.source || undefined,
      has_benchmarks: filters.has_benchmarks || undefined,
      tool_calling: filters.tool_calling || undefined,
      is_moe: filters.is_moe || undefined,
      vision: filters.vision || undefined,
      min_params: sb?.min,
      max_params: sb?.max,
      sort: filters.sort,
    }
  }, [debouncedQ, filters])

  // Refetch when filters change (offset=0)
  useEffect(() => {
    const seq = ++requestSeq.current
    setLoading(true)
    setError(null)
    listModels({ ...queryParams, limit: PAGE_SIZE, offset: 0 })
      .then(r => {
        if (seq !== requestSeq.current) return
        setModels(r.models)
        setTotal(r.total)
        setFacets(r.facets)
      })
      .catch(e => {
        if (seq !== requestSeq.current) return
        setError(e instanceof Error ? e.message : 'Failed to load models')
      })
      .finally(() => {
        if (seq === requestSeq.current) setLoading(false)
      })
  }, [queryParams])

  const loadMore = async () => {
    setLoadingMore(true)
    try {
      const r = await listModels({ ...queryParams, limit: PAGE_SIZE, offset: models.length })
      setModels(prev => [...prev, ...r.models])
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load more')
    } finally {
      setLoadingMore(false)
    }
  }

  const hasMore = models.length < total
  const activeFilterCount = [
    filters.has_benchmarks, filters.tool_calling, filters.is_moe, filters.vision,
    filters.size !== 'any', filters.family, filters.source,
  ].filter(Boolean).length

  return (
    <div className="space-y-5">
      {/* Sticky header — search + sort */}
      <div className="sticky top-16 -mx-4 px-4 py-3 bg-background/95 backdrop-blur z-30 border-b border-border/40">
        <div className="flex flex-col sm:flex-row gap-3">
          <div className="relative flex-1">
            <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search 3,000+ models by name or family…"
              value={filters.q}
              onChange={e => setFilters(f => ({ ...f, q: e.target.value }))}
              className="input pl-10 pr-10"
            />
            {filters.q && (
              <button
                onClick={() => setFilters(f => ({ ...f, q: '' }))}
                className="absolute right-2 top-1/2 -translate-y-1/2 p-1 text-muted-foreground hover:text-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            )}
          </div>
          <select
            value={filters.sort}
            onChange={e => setFilters(f => ({ ...f, sort: e.target.value as SortKey }))}
            className="input sm:max-w-[180px]"
          >
            {SORT_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </div>
        <div className="mt-2 flex items-center justify-between text-xs text-muted-foreground">
          <span>
            {loading ? 'Searching…' : `${total.toLocaleString()} model${total === 1 ? '' : 's'}`}
            {activeFilterCount > 0 && (
              <button
                onClick={() => setFilters(defaultFilters)}
                className="ml-3 text-primary hover:underline"
              >
                Clear filters ({activeFilterCount})
              </button>
            )}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[220px_1fr] gap-6">
        {/* Sidebar */}
        <aside className="space-y-5 md:sticky md:top-44 md:self-start md:max-h-[calc(100vh-12rem)] md:overflow-y-auto md:pr-2">
          <FilterSection title="Capabilities">
            <Toggle
              checked={filters.has_benchmarks}
              onChange={v => setFilters(f => ({ ...f, has_benchmarks: v }))}
              label="Has benchmark scores"
              count={facets.total_with_benchmarks}
            />
            <Toggle
              checked={filters.tool_calling}
              onChange={v => setFilters(f => ({ ...f, tool_calling: v }))}
              label="Tool calling"
              count={facets.total_tool_calling}
            />
            <Toggle
              checked={filters.is_moe}
              onChange={v => setFilters(f => ({ ...f, is_moe: v }))}
              label="Mixture of Experts"
              count={facets.total_moe}
            />
            <Toggle
              checked={filters.vision}
              onChange={v => setFilters(f => ({ ...f, vision: v }))}
              label="Vision / multimodal"
              count={facets.total_vision}
            />
          </FilterSection>

          <FilterSection title="Size">
            <div className="space-y-1">
              {SIZE_BUCKETS.map(b => (
                <Radio
                  key={b.id}
                  checked={filters.size === b.id}
                  onChange={() => setFilters(f => ({ ...f, size: b.id }))}
                  label={b.label}
                />
              ))}
            </div>
          </FilterSection>

          <FilterSection title="Available in">
            <div className="space-y-1">
              <Radio
                checked={!filters.source}
                onChange={() => setFilters(f => ({ ...f, source: null }))}
                label="Any source"
              />
              {SOURCE_OPTIONS.map(s => (
                <Radio
                  key={s.id}
                  checked={filters.source === s.id}
                  onChange={() => setFilters(f => ({ ...f, source: s.id }))}
                  label={s.label}
                  count={facets.sources.find(f => f.value === s.id)?.count}
                />
              ))}
            </div>
          </FilterSection>

          <FilterSection title="Family" defaultCollapsed>
            <div className="space-y-1">
              <Radio
                checked={!filters.family}
                onChange={() => setFilters(f => ({ ...f, family: null }))}
                label="All families"
              />
              {facets.families.slice(0, 20).map(f => (
                <Radio
                  key={f.value}
                  checked={filters.family === f.value}
                  onChange={() => setFilters(s => ({ ...s, family: f.value }))}
                  label={f.value}
                  count={f.count}
                />
              ))}
            </div>
          </FilterSection>
        </aside>

        {/* Results */}
        <main className="min-w-0">
          {error && (
            <div className="card border-destructive/30 bg-destructive/5 text-destructive mb-4">{error}</div>
          )}

          {loading && models.length === 0 ? (
            <div className="flex justify-center py-20"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>
          ) : models.length === 0 ? (
            <div className="card text-center text-muted-foreground py-12">
              No models match these filters. Try clearing some, or check the
              <Link to="/sources" className="text-primary hover:underline ml-1">data sources</Link>.
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
                {models.map(m => <ModelCard key={m.canonical_id} model={m} />)}
              </div>

              {hasMore && (
                <div className="text-center mt-6">
                  <button
                    onClick={loadMore}
                    disabled={loadingMore}
                    className="btn-outline gap-2"
                  >
                    {loadingMore ? (
                      <><Loader2 className="h-4 w-4 animate-spin" /> Loading…</>
                    ) : (
                      <>Load more ({total - models.length} remaining)</>
                    )}
                  </button>
                </div>
              )}
            </>
          )}
        </main>
      </div>
    </div>
  )
}


function FilterSection({ title, children, defaultCollapsed = false }:
  { title: string; children: React.ReactNode; defaultCollapsed?: boolean }) {
  const [open, setOpen] = useState(!defaultCollapsed)
  return (
    <div className="space-y-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center justify-between w-full text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground"
      >
        <span>{title}</span>
        {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
      </button>
      {open && <div className="space-y-1">{children}</div>}
    </div>
  )
}

function Toggle({ checked, onChange, label, count }:
  { checked: boolean; onChange: (v: boolean) => void; label: string; count?: number }) {
  return (
    <label className="flex items-center justify-between gap-2 text-sm cursor-pointer hover:text-foreground text-muted-foreground py-0.5">
      <span className="flex items-center gap-2">
        <input
          type="checkbox"
          checked={checked}
          onChange={e => onChange(e.target.checked)}
          className="rounded border-border"
        />
        {label}
      </span>
      {count !== undefined && count > 0 && (
        <span className="text-xs opacity-50">{count}</span>
      )}
    </label>
  )
}

function Radio({ checked, onChange, label, count }:
  { checked: boolean; onChange: () => void; label: string; count?: number }) {
  return (
    <label className="flex items-center justify-between gap-2 text-sm cursor-pointer hover:text-foreground py-0.5">
      <span className={cn('flex items-center gap-2', checked ? 'text-foreground' : 'text-muted-foreground')}>
        <input
          type="radio"
          checked={checked}
          onChange={onChange}
          className="border-border"
        />
        {label}
      </span>
      {count !== undefined && count > 0 && (
        <span className="text-xs opacity-50">{count}</span>
      )}
    </label>
  )
}


function ModelCard({ model }: { model: ModelSummary }) {
  const sourceBadges = model.artifacts.length > 0
    ? Array.from(new Set(model.artifacts.map(a => sourceLabel(a.source))))
    : []

  const ctxLabel = model.context_length > 0
    ? `${model.context_length >= 1000 ? Math.round(model.context_length / 1000) + 'K' : model.context_length} ctx`
    : null

  const sizeLabel = model.is_moe && model.active_params_b > 0 && model.total_params_b > 0
    ? `${model.total_params_b}B / ${model.active_params_b}B active`
    : model.total_params_b > 0
      ? `${model.total_params_b}B`
      : model.parameter_size

  return (
    <Link to={`/model/${encodeURIComponent(model.canonical_id)}`} className="card-hover">
      <div className="flex items-start justify-between gap-3">
        <div className="space-y-1.5 min-w-0 flex-1">
          <div className="flex items-center gap-1.5 flex-wrap">
            <h3 className="font-semibold truncate">{model.display_name}</h3>
            {model.is_moe && <span className="badge badge-accent text-[10px]">MoE</span>}
            {model.vision && <span className="badge badge-accent text-[10px]">vision</span>}
            {model.tool_calling && <span className="badge badge-primary text-[10px]">tools</span>}
          </div>

          <div className="flex items-center gap-x-3 gap-y-0.5 text-xs text-muted-foreground flex-wrap">
            <span className="font-mono">{model.family}</span>
            <span>·</span>
            <span>{sizeLabel}</span>
            {ctxLabel && <><span>·</span><span>{ctxLabel}</span></>}
          </div>

          <div className="flex items-center gap-1.5 flex-wrap pt-1">
            {sourceBadges.map(s => (
              <span key={s} className="badge badge-secondary text-[10px]">{s}</span>
            ))}
            {model.benchmark_count > 0 && (
              <span className="text-[10px] text-muted-foreground">{model.benchmark_count} benchmarks</span>
            )}
          </div>
        </div>

        {model.artifacts[0]?.size_mb > 0 && (
          <span className="text-xs text-muted-foreground shrink-0 whitespace-nowrap">
            {formatSize(model.artifacts[0].size_mb)}
          </span>
        )}
      </div>
    </Link>
  )
}

function sourceLabel(source: string): string {
  if (source === 'ollama') return 'Ollama'
  if (source === 'lmstudio-community') return 'LM Studio'
  if (source === 'huggingface_gguf') return 'HF GGUF'
  return source
}

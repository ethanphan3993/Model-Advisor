import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import {
  Brain, Cpu, Zap, HardDrive, Loader2, Database, AlertCircle, RefreshCw,
  Info, ChevronDown, ChevronUp, Star, AlertTriangle, Eye,
} from 'lucide-react'
import { useScan } from '../hooks/useScan'
import { useMeta } from '../hooks/useMeta'
import { useRecommend } from '../hooks/useRecommend'
import { listModels, triggerRefresh } from '../lib/api'
import { ScoreBar } from '../components/ScoreBar'
import { InstallCommand } from '../components/InstallCommand'
import { cn, formatScore, formatSize, scoreBg, scoreColor } from '../lib/utils'
import type { Recommendation } from '../types'

type SortKey = 'fit' | 'speed' | 'smallest' | 'benchmarks' | 'reasoning'

const SORT_OPTIONS: { value: SortKey; label: string }[] = [
  { value: 'fit', label: 'Best overall fit' },
  { value: 'benchmarks', label: 'Highest benchmarks' },
  { value: 'speed', label: 'Fastest decode' },
  { value: 'smallest', label: 'Smallest footprint' },
  { value: 'reasoning', label: 'Best reasoning' },
]

function applySort(recs: Recommendation[], sort: SortKey): Recommendation[] {
  const sorted = [...recs]
  const tps = (r: Recommendation) => (r.estimated_tokens_per_sec[0] + r.estimated_tokens_per_sec[1]) / 2
  switch (sort) {
    case 'fit': sorted.sort((a, b) => b.fit_score - a.fit_score); break
    case 'benchmarks': sorted.sort((a, b) => b.use_case_score - a.use_case_score); break
    case 'speed': sorted.sort((a, b) => tps(b) - tps(a)); break
    case 'smallest':
      sorted.sort((a, b) =>
        (a.estimated_size_mb + a.estimated_kv_cache_mb) -
        (b.estimated_size_mb + b.estimated_kv_cache_mb)); break
    case 'reasoning': {
      const r = (rec: Recommendation) => {
        const gpqa = rec.provenance.use_case_components.find(e => e.benchmark === 'gpqa')?.normalized || 0
        const bbh = rec.provenance.use_case_components.find(e => e.benchmark === 'bbh')?.normalized || 0
        return gpqa + bbh
      }
      sorted.sort((a, b) => r(b) - r(a)); break
    }
  }
  return sorted
}

export default function Home() {
  const [params, setParams] = useSearchParams()
  const { data, loading: scanLoading, scan } = useScan()
  const { useCases, harnesses } = useMeta()
  const { recommendations, hardware, loading: recoLoading, error, fetch } = useRecommend()

  // Filter state — synced to URL so links are shareable.
  const useCase = params.get('use_case') || 'assistant'
  const harness = params.get('harness') || ''
  const [sort, setSort] = useState<SortKey>('fit')
  const [includeBig, setIncludeBig] = useState(false)
  const [sourceFilter, setSourceFilter] = useState<Set<string>>(new Set())
  const [showExplainer, setShowExplainer] = useState(false)

  // First-run / staleness detection
  const [catalogState, setCatalogState] = useState<'checking' | 'empty' | 'stale' | 'fresh'>('checking')
  const [refreshing, setRefreshing] = useState(false)

  useEffect(() => { if (!data && !scanLoading) scan() }, []) // eslint-disable-line
  useEffect(() => {
    listModels({ has_benchmarks: true, limit: 1 })
      .then(r => setCatalogState(r.total === 0 ? 'empty' : r.total < 100 ? 'stale' : 'fresh'))
      .catch(() => setCatalogState('empty'))
  }, [])

  // Refetch recommendations whenever filters change
  useEffect(() => {
    if (useCase) fetch({ use_case: useCase, harness: harness || null, limit: 15, include_too_big: includeBig })
  }, [useCase, harness, includeBig, fetch])

  const handleRefresh = async () => {
    setRefreshing(true)
    try {
      await triggerRefresh()
      const r = await listModels({ has_benchmarks: true, limit: 1 })
      setCatalogState(r.total === 0 ? 'empty' : r.total < 100 ? 'stale' : 'fresh')
      fetch({ use_case: useCase, harness: harness || null, limit: 15, include_too_big: includeBig })
    } finally { setRefreshing(false) }
  }

  const setParam = (key: string, value: string | null) => {
    const next = new URLSearchParams(params)
    if (value) next.set(key, value); else next.delete(key)
    setParams(next, { replace: true })
  }

  const filteredRecs = useMemo(() => {
    if (sourceFilter.size === 0) return recommendations
    return recommendations.filter(r =>
      r.install_options.some(opt => {
        const s = opt.source
        if (sourceFilter.has('ollama') && s === 'ollama') return true
        if (sourceFilter.has('lmstudio') && (s === 'lmstudio-community' || s === 'huggingface_gguf')) return true
        if (sourceFilter.has('huggingface') && (s === 'huggingface_gguf' || s === 'lmstudio-community')) return true
        return false
      })
    )
  }, [recommendations, sourceFilter])

  const sortedRecs = useMemo(() => applySort(filteredRecs, sort), [filteredRecs, sort])

  const toggleSource = (s: string) =>
    setSourceFilter(prev => {
      const next = new Set(prev)
      next.has(s) ? next.delete(s) : next.add(s)
      return next
    })

  return (
    <div className="space-y-6">
      {/* Hero */}
      <div className="text-center space-y-3 py-4">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
          <Brain className="h-7 w-7 text-primary" />
        </div>
        <h1 className="text-3xl font-bold tracking-tight">Model Advisor</h1>
        <p className="text-sm text-muted-foreground max-w-2xl mx-auto">
          Hardware-aware ranking of local LLMs for your Mac — MoE-savvy, bandwidth-bound,
          with full benchmark provenance.
        </p>
      </div>

      {/* Catalog warning */}
      {(catalogState === 'empty' || catalogState === 'stale') && (
        <div className={cn('card border flex items-start gap-3',
          catalogState === 'empty' ? 'border-yellow-400/40 bg-yellow-400/5' : 'border-primary/30 bg-primary/5')}>
          {catalogState === 'empty'
            ? <AlertCircle className="h-5 w-5 text-yellow-400 shrink-0 mt-0.5" />
            : <Database className="h-5 w-5 text-primary shrink-0 mt-0.5" />}
          <div className="flex-1 space-y-2">
            <h3 className="font-semibold text-sm">
              {catalogState === 'empty' ? 'No catalog data yet' : 'Catalog looks light'}
            </h3>
            <button onClick={handleRefresh} disabled={refreshing} className="btn-primary gap-2 text-sm">
              {refreshing ? (<><Loader2 className="h-4 w-4 animate-spin" /> Refreshing…</>)
                          : (<><RefreshCw className="h-4 w-4" /> Populate catalog</>)}
            </button>
          </div>
        </div>
      )}

      {/* Hardware tile */}
      <div className="card border-border/50">
        {scanLoading && (
          <div className="flex items-center gap-3 text-muted-foreground text-sm">
            <Loader2 className="h-4 w-4 animate-spin" /> Scanning hardware…
          </div>
        )}
        {data && (
          <>
            <div className="flex items-start justify-between gap-6">
              <div className="space-y-1 min-w-0">
                <div className="text-xs text-muted-foreground">Detected device</div>
                <h2 className="text-lg font-semibold truncate">{data.hardware.model}</h2>
                <p className="text-xs text-muted-foreground">{data.ai_capability.interpretation}</p>
              </div>
              <div className={cn('text-right shrink-0', scoreColor(data.ai_capability.composite_score))}>
                <div className="text-2xl font-bold">{formatScore(data.ai_capability.composite_score)}</div>
                <div className="text-[10px] text-muted-foreground">capability</div>
              </div>
            </div>
            <div className="grid grid-cols-3 gap-3 mt-4">
              <Stat icon={Cpu} top={data.hardware.chip.chip} bottom={`${data.hardware.chip.gpu_cores} GPU cores`} />
              <Stat icon={Zap} top={`${data.memory.total_gb} GB RAM`} bottom={`${data.memory.available_gb.toFixed(1)} GB free`} />
              <Stat icon={HardDrive} top={`${data.hardware.chip.neural_engine_cores}-core NPU`} bottom={data.hardware.chip.generation} />
            </div>
          </>
        )}
      </div>

      {/* Filters: use case + harness + source + sort */}
      <div className="card space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div>
            <label className="text-xs text-muted-foreground block mb-1">For what task?</label>
            <select value={useCase} onChange={e => setParam('use_case', e.target.value)} className="input h-9 w-full text-sm">
              {useCases.map(u => <option key={u.id} value={u.id}>{u.name}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Agent / runtime (optional)</label>
            <select value={harness} onChange={e => setParam('harness', e.target.value || null)} className="input h-9 w-full text-sm">
              <option value="">Any harness</option>
              {harnesses.map(h => <option key={h.id} value={h.id}>{h.name}</option>)}
            </select>
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Sort by</label>
            <select value={sort} onChange={e => setSort(e.target.value as SortKey)} className="input h-9 w-full text-sm">
              {SORT_OPTIONS.map(o => <option key={o.value} value={o.value}>{o.label}</option>)}
            </select>
          </div>
        </div>

        <div className="flex items-center gap-3 flex-wrap pt-1 border-t border-border/30">
          <span className="text-xs text-muted-foreground">Available in:</span>
          {[
            { id: 'ollama', label: 'Ollama' },
            { id: 'lmstudio', label: 'LM Studio' },
            { id: 'huggingface', label: 'HuggingFace' },
          ].map(s => (
            <button
              key={s.id}
              onClick={() => toggleSource(s.id)}
              className={cn('badge px-2.5 py-1 text-xs cursor-pointer transition-colors',
                sourceFilter.has(s.id)
                  ? 'bg-primary/15 text-primary border border-primary/40'
                  : 'bg-secondary text-muted-foreground hover:bg-secondary/80 border border-transparent')}
            >{s.label}</button>
          ))}
          <span className="text-muted-foreground/40 text-xs">·</span>
          <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
            <input type="checkbox" checked={includeBig} onChange={e => setIncludeBig(e.target.checked)} className="rounded border-border" />
            Include too-big models
          </label>
          <button
            onClick={() => setShowExplainer(!showExplainer)}
            className="ml-auto inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-primary"
          >
            <Info className="h-3.5 w-3.5" />
            How is this scored?
          </button>
        </div>

        {showExplainer && <ScoreExplainer onClose={() => setShowExplainer(false)} />}
      </div>

      {/* Result count */}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>
          {recoLoading ? 'Computing…' :
            sourceFilter.size > 0
              ? `${filteredRecs.length} of ${recommendations.length} match filter`
              : `${recommendations.length} ranked picks`}
        </span>
        {error && <span className="text-destructive">{error}</span>}
      </div>

      {/* Empty / loading states */}
      {recoLoading && recommendations.length === 0 ? (
        <div className="flex justify-center py-10"><Loader2 className="h-7 w-7 animate-spin text-muted-foreground" /></div>
      ) : sortedRecs.length === 0 && recommendations.length > 0 && sourceFilter.size > 0 ? (
        <div className="card text-center py-8 space-y-2">
          <p className="text-sm">No top picks available in your selected sources.</p>
          <button onClick={() => setSourceFilter(new Set())} className="btn-outline text-sm">Clear source filter</button>
        </div>
      ) : recommendations.length === 0 ? (
        <div className="card text-center py-8 text-sm text-muted-foreground">
          No models match these filters. Try a different use case or remove the harness restriction.
        </div>
      ) : (
        <div className="space-y-3">
          {sortedRecs.map((r, i) => (
            <RecCard
              key={r.canonical_id}
              rec={r}
              position={i + 1}
              highlight={i === 0}
              sort={sort}
              top={sortedRecs[0]}
              hardware={hardware}
            />
          ))}
        </div>
      )}

      {sortedRecs.length >= 2 && (
        <div className="text-center pt-2">
          <Link
            to={`/compare?a=${sortedRecs[0].canonical_id}&b=${sortedRecs[1].canonical_id}`}
            className="btn-outline gap-2 inline-flex text-sm"
          >
            <Eye className="h-4 w-4" /> Compare top 2 side-by-side
          </Link>
        </div>
      )}
    </div>
  )
}

function Stat({ icon: Icon, top, bottom }: { icon: React.ElementType; top: string; bottom: string }) {
  return (
    <div className="flex items-center gap-2 p-2 rounded-lg bg-secondary/40 min-w-0">
      <Icon className="h-4 w-4 text-primary shrink-0" />
      <div className="min-w-0">
        <div className="text-xs font-medium truncate">{top}</div>
        <div className="text-[10px] text-muted-foreground truncate">{bottom}</div>
      </div>
    </div>
  )
}

function RecCard({ rec, position, highlight, sort, top, hardware }: {
  rec: Recommendation; position: number; highlight?: boolean; sort: SortKey
  top: Recommendation
  hardware: { total_memory_gb: number; available_memory_gb: number; chip: string } | null
}) {
  const [expanded, setExpanded] = useState(false)
  const headlineValue = sort === 'speed'
    ? Math.round((rec.estimated_tokens_per_sec[0] + rec.estimated_tokens_per_sec[1]) / 2)
    : sort === 'smallest'
      ? Math.round((rec.estimated_size_mb + rec.estimated_kv_cache_mb) / 1024 * 10) / 10
      : sort === 'benchmarks' ? rec.use_case_score : rec.fit_score
  const headlineSuffix = sort === 'speed' ? 'tps' : sort === 'smallest' ? 'GB' : ''

  const totalGb = hardware?.total_memory_gb ?? 0
  const residentGb = (rec.estimated_size_mb + rec.estimated_kv_cache_mb) / 1024
  const ramPct = totalGb > 0 ? (residentGb / totalGb) * 100 : 0
  const avgTps = (rec.estimated_tokens_per_sec[0] + rec.estimated_tokens_per_sec[1]) / 2

  const showComparison = position > 1 && top && top.canonical_id !== rec.canonical_id
  const comparisonLine = showComparison ? buildComparison(rec, top) : null

  const confColor = rec.confidence_pct >= 80 ? 'bg-accent/15 text-accent'
    : rec.confidence_pct >= 50 ? 'bg-primary/15 text-primary' : 'bg-yellow-400/15 text-yellow-400'

  return (
    <div className={cn('card border', highlight && scoreBg(rec.fit_score))}>
      <div className="flex items-start gap-4">
        <div className={cn('flex h-14 w-14 flex-col items-center justify-center rounded-xl shrink-0', scoreColor(rec.fit_score))}>
          <span className="text-xl font-bold leading-none">{Number(headlineValue).toFixed(headlineSuffix === '' ? 1 : 0)}</span>
          {headlineSuffix && <span className="text-[10px] mt-0.5 opacity-70">{headlineSuffix}</span>}
        </div>
        <div className="flex-1 min-w-0 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="badge badge-secondary text-[11px]">#{position}</span>
            {highlight && sort === 'fit' && <Star className="h-4 w-4 text-yellow-400 fill-yellow-400" />}
            <Link to={`/model/${encodeURIComponent(rec.canonical_id)}`} className="font-semibold hover:text-primary transition-colors">
              {rec.display_name}
            </Link>
            <span className="badge badge-secondary text-[11px]">{rec.parameter_size}</span>
            {rec.is_moe && <span className="badge badge-accent text-[11px]">MoE</span>}
            <span className={cn('badge text-[11px]', confColor)} title={`${rec.benchmarks_measured} of ${rec.benchmarks_expected} benchmarks`}>
              {rec.confidence_pct}% confidence
            </span>
          </div>
          <p className="text-sm text-muted-foreground">{rec.why}</p>
          {comparisonLine && (
            <p className="text-xs text-muted-foreground italic">
              <span className="opacity-70">vs #1:</span> {comparisonLine}
            </p>
          )}
          <div className="grid grid-cols-3 gap-3 pt-1">
            <ScoreBar label="Use case fit" score={Math.min(10, rec.use_case_score)} />
            <ScoreBar label="Hardware fit" score={rec.hardware_fit} />
            <ScoreBar label="Harness fit" score={rec.harness_fit} />
          </div>
          <div className="text-xs space-y-1 pt-1">
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="font-mono opacity-70">RAM</span>
              <span>
                {formatSize(rec.estimated_size_mb)} weights
                {rec.estimated_kv_cache_mb > 0 && ` + ${formatSize(rec.estimated_kv_cache_mb)} KV`}
                {' = '}<strong className="text-foreground">{residentGb.toFixed(1)} GB</strong>
                {totalGb > 0 && <span className="opacity-70"> / {totalGb} GB ({ramPct.toFixed(0)}%)</span>}
              </span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="font-mono opacity-70">TPS</span>
              <span>
                {rec.bandwidth_gb_s} GB/s ÷ {rec.active_params_b}B {rec.is_moe && 'active'} × 0.70 ≈
                {' '}<strong className="text-foreground">{Math.round(avgTps)} tok/s</strong>
              </span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="font-mono opacity-70">QNT</span>
              <span>
                <strong className="text-foreground">{rec.quantization_recommended}</strong>
                {rec.quant_quality_factor < 1.0 && <span className="opacity-70"> · ~{Math.round(rec.quant_quality_factor * 100)}% of FP16 quality</span>}
              </span>
            </div>
          </div>
          {rec.warnings.length > 0 && (
            <div className="flex items-start gap-1 text-xs text-yellow-400 pt-1">
              <AlertTriangle className="h-3 w-3 mt-0.5 shrink-0" />
              <span>{rec.warnings.join(' · ')}</span>
            </div>
          )}
        </div>
        <button onClick={() => setExpanded(!expanded)} className="btn-outline h-8 px-2 text-xs gap-1 shrink-0">
          {expanded ? 'Less' : 'Details'}
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        </button>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t space-y-4">
          {rec.provenance.use_case_components.length > 0 && (
            <div className="space-y-1">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase">Benchmark evidence</h4>
              {rec.provenance.use_case_components.map(e => (
                <div key={e.benchmark} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">{e.benchmark} <span className="opacity-60">via {e.source}</span></span>
                  <span className="font-mono">{e.value.toFixed(1)} <span className="opacity-60">({(e.normalized * 100).toFixed(0)}%)</span></span>
                </div>
              ))}
            </div>
          )}
          {rec.install_options.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase">Install</h4>
              {rec.install_options.map((opt, i) => <InstallCommand key={i} option={opt} />)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function buildComparison(rec: Recommendation, top: Recommendation): string {
  const parts: string[] = []
  const fitGap = top.fit_score - rec.fit_score
  if (fitGap > 0.05) parts.push(`-${fitGap.toFixed(1)} fit`)
  const recTps = (rec.estimated_tokens_per_sec[0] + rec.estimated_tokens_per_sec[1]) / 2
  const topTps = (top.estimated_tokens_per_sec[0] + top.estimated_tokens_per_sec[1]) / 2
  const tpsRatio = recTps / Math.max(topTps, 1)
  if (tpsRatio > 1.3) parts.push(`${Math.round(recTps)} vs ${Math.round(topTps)} tok/s (faster)`)
  else if (tpsRatio < 0.77) parts.push(`${Math.round(recTps)} vs ${Math.round(topTps)} tok/s (slower)`)
  const sizeDelta = (rec.estimated_size_mb + rec.estimated_kv_cache_mb) / 1024 -
                    (top.estimated_size_mb + top.estimated_kv_cache_mb) / 1024
  if (Math.abs(sizeDelta) > 1) parts.push(sizeDelta > 0 ? `+${sizeDelta.toFixed(1)} GB RAM` : `${sizeDelta.toFixed(1)} GB RAM`)
  if (rec.confidence_pct < top.confidence_pct - 10) parts.push(`${rec.confidence_pct}% vs ${top.confidence_pct}% confidence`)
  return parts.length > 0 ? parts.join(', ') : 'similar profile'
}

function ScoreExplainer({ onClose }: { onClose: () => void }) {
  return (
    <div className="bg-secondary/30 border border-border/60 rounded-md p-3 text-xs space-y-2">
      <div className="flex items-start justify-between">
        <h3 className="font-semibold">How the scores work</h3>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground">close</button>
      </div>
      <pre className="font-mono text-[11px] bg-background/60 rounded p-2 overflow-x-auto">
{`fit_score = 0.55 × use_case_score    (benchmark match)
          + 0.30 × hardware_fit       (RAM + decode speed)
          + 0.15 × harness_fit        (compatibility)`}
      </pre>
      <div className="text-muted-foreground">
        Confidence shows what fraction of the use case's expected benchmarks we have measurements for.
      </div>
    </div>
  )
}

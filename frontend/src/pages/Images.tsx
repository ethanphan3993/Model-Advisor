import { useEffect, useMemo, useState } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  Image as ImageIcon, Loader2, AlertTriangle, ChevronDown, ChevronUp,
  Cpu, Zap, Star, Info, ExternalLink, Copy, Check,
} from 'lucide-react'
import {
  listImageUseCases, listImageHarnesses, recommendImages, getImageCatalog,
} from '../lib/api'
import { ScoreBar } from '../components/ScoreBar'
import { cn, copyToClipboard, scoreBg, scoreColor } from '../lib/utils'
import type {
  ImageHarness, ImageHardwareSnapshot, ImageInstallOption, ImageModelCard,
  ImageRecommendation, ImageUseCase,
} from '../types'


type ViewMode = 'ranked' | 'catalog'

export default function Images() {
  const [params, setParams] = useSearchParams()
  const [useCases, setUseCases] = useState<ImageUseCase[]>([])
  const [harnesses, setHarnesses] = useState<ImageHarness[]>([])
  const [recs, setRecs] = useState<ImageRecommendation[]>([])
  const [catalog, setCatalog] = useState<ImageModelCard[]>([])
  const [hardware, setHardware] = useState<ImageHardwareSnapshot | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [showExplainer, setShowExplainer] = useState(false)
  const [includeBig, setIncludeBig] = useState(false)

  const view = (params.get('view') as ViewMode) || 'ranked'
  const useCase = params.get('use_case') || 'image_generation'
  const harness = params.get('harness') || ''

  // Load metadata + full catalog once.
  useEffect(() => {
    Promise.all([listImageUseCases(), listImageHarnesses(), getImageCatalog()])
      .then(([u, h, c]) => { setUseCases(u); setHarnesses(h); setCatalog(c) })
      .catch(e => setError(String(e)))
  }, [])

  // Refetch recommendations on filter change. Only when in 'ranked' view.
  useEffect(() => {
    if (view !== 'ranked') return
    setLoading(true); setError(null)
    recommendImages({ use_case: useCase, harness: harness || null, limit: 15, include_too_big: includeBig })
      .then(r => { setRecs(r.recommendations); setHardware(r.hardware_snapshot) })
      .catch(e => setError(String(e)))
      .finally(() => setLoading(false))
  }, [view, useCase, harness, includeBig])

  const setParam = (key: string, value: string | null) => {
    const next = new URLSearchParams(params)
    if (value) next.set(key, value); else next.delete(key)
    setParams(next, { replace: true })
  }

  const sortedRecs = useMemo(() => [...recs].sort((a, b) => b.fit_score - a.fit_score), [recs])

  return (
    <div className="space-y-6">
      <div className="text-center space-y-3 py-4">
        <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-primary/10">
          <ImageIcon className="h-7 w-7 text-primary" />
        </div>
        <h1 className="text-3xl font-bold tracking-tight">Image Generation</h1>
        <p className="text-sm text-muted-foreground max-w-2xl mx-auto">
          Local diffusion &amp; flow-matching models — FLUX, SD 3.5, SDXL, HiDream, AuraFlow.
          Compute-bound time-per-image cost model, not text-LLM TPS.
        </p>
        <p className="text-xs text-muted-foreground max-w-2xl mx-auto">
          v0 catalog: {' '}
          <span className="opacity-70">~20 hand-curated models, scores from published model cards & GenEval / imagen-leaderboard.dev. Hardware estimates from M3 Max / M4 Max references scaled by chip TFLOPS.</span>
        </p>
      </div>

      {/* View toggle */}
      <div className="flex items-center gap-2 text-xs">
        <button
          onClick={() => setParam('view', null)}
          className={cn('px-3 py-1.5 rounded-md transition-colors',
            view === 'ranked' ? 'bg-primary/15 text-primary border border-primary/40' : 'bg-secondary text-muted-foreground hover:bg-secondary/80 border border-transparent')}
        >Ranked picks for your Mac</button>
        <button
          onClick={() => setParam('view', 'catalog')}
          className={cn('px-3 py-1.5 rounded-md transition-colors',
            view === 'catalog' ? 'bg-primary/15 text-primary border border-primary/40' : 'bg-secondary text-muted-foreground hover:bg-secondary/80 border border-transparent')}
        >Browse all {catalog.length} models</button>
      </div>

      {/* Hardware tile */}
      {hardware && view === 'ranked' && (
        <div className="card border-border/50">
          <div className="flex items-center justify-between gap-4 flex-wrap">
            <div className="flex items-center gap-3 min-w-0">
              <Cpu className="h-5 w-5 text-primary shrink-0" />
              <div className="min-w-0">
                <div className="font-semibold text-sm truncate">{hardware.chip}</div>
                <div className="text-xs text-muted-foreground">
                  {hardware.fp16_tflops.toFixed(1)} TFLOPS FP16 · {hardware.total_memory_gb} GB RAM ({hardware.available_memory_gb.toFixed(1)} GB free)
                </div>
              </div>
            </div>
            <div className="text-right">
              <div className="flex items-center gap-1 text-xs text-muted-foreground">
                <Zap className="h-3 w-3" />
                Compute-bound
              </div>
              <div className="text-[10px] text-muted-foreground">image gen, not bandwidth-bound</div>
            </div>
          </div>
        </div>
      )}

      {view === 'catalog' ? (
        <ImageCatalogGrid catalog={catalog} harnesses={harnesses} />
      ) : (<>

      {/* Filters */}
      <div className="card space-y-3">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-muted-foreground block mb-1">For what task?</label>
            <select value={useCase} onChange={e => setParam('use_case', e.target.value)} className="input h-9 w-full text-sm">
              {useCases.map(u => <option key={u.id} value={u.id}>{u.name}</option>)}
            </select>
            {useCases.find(u => u.id === useCase) && (
              <p className="text-[11px] text-muted-foreground mt-1">{useCases.find(u => u.id === useCase)!.tagline}</p>
            )}
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Image app / runtime (optional)</label>
            <select value={harness} onChange={e => setParam('harness', e.target.value || null)} className="input h-9 w-full text-sm">
              <option value="">Any</option>
              {harnesses.map(h => <option key={h.id} value={h.id}>{h.name}</option>)}
            </select>
            {harnesses.find(h => h.id === harness) && (
              <p className="text-[11px] text-muted-foreground mt-1">{harnesses.find(h => h.id === harness)!.description}</p>
            )}
          </div>
        </div>

        <div className="flex items-center gap-3 flex-wrap pt-1 border-t border-border/30">
          <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer">
            <input type="checkbox" checked={includeBig} onChange={e => setIncludeBig(e.target.checked)} className="rounded border-border" />
            Include models that don't fit
          </label>
          <button
            onClick={() => setShowExplainer(!showExplainer)}
            className="ml-auto inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-primary"
          >
            <Info className="h-3.5 w-3.5" />
            How is this scored?
          </button>
        </div>

        {showExplainer && <ImageScoreExplainer onClose={() => setShowExplainer(false)} />}
      </div>

      {/* Status row */}
      <div className="flex items-center justify-between text-xs text-muted-foreground">
        <span>{loading ? 'Computing…' : `${sortedRecs.length} ranked picks`}</span>
        {error && <span className="text-destructive">{error}</span>}
      </div>

      {/* Results */}
      {loading && sortedRecs.length === 0 ? (
        <div className="flex justify-center py-10"><Loader2 className="h-7 w-7 animate-spin text-muted-foreground" /></div>
      ) : sortedRecs.length === 0 ? (
        <div className="card text-center py-8 text-sm text-muted-foreground">
          No models match these filters. Try a different use case or remove the harness restriction.
        </div>
      ) : (
        <div className="space-y-3">
          {sortedRecs.map((r, i) => (
            <ImageRecCard key={r.canonical_id} rec={r} highlight={i === 0} top={sortedRecs[0]} />
          ))}
        </div>
      )}
      </>)}
    </div>
  )
}


function ImageCatalogGrid({ catalog, harnesses }: { catalog: ImageModelCard[]; harnesses: ImageHarness[] }) {
  const [familyFilter, setFamilyFilter] = useState<string>('')
  const [supportsFilter, setSupportsFilter] = useState<string>('')

  const families = useMemo(
    () => Array.from(new Set(catalog.map(m => m.family))).sort(),
    [catalog],
  )

  const filtered = useMemo(() => {
    return catalog.filter(m => {
      if (familyFilter && m.family !== familyFilter) return false
      if (supportsFilter && !m.supports.includes(supportsFilter)) return false
      return true
    })
  }, [catalog, familyFilter, supportsFilter])

  const harnessName = (id: string) => harnesses.find(h => h.id === id)?.name || id

  return (
    <div className="space-y-4">
      <div className="card">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Family</label>
            <select value={familyFilter} onChange={e => setFamilyFilter(e.target.value)} className="input h-9 w-full text-sm">
              <option value="">All families ({catalog.length})</option>
              {families.map(f => (
                <option key={f} value={f}>{f} ({catalog.filter(m => m.family === f).length})</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Capability</label>
            <select value={supportsFilter} onChange={e => setSupportsFilter(e.target.value)} className="input h-9 w-full text-sm">
              <option value="">Any</option>
              <option value="image_generation">Generation</option>
              <option value="image_editing">Editing</option>
            </select>
          </div>
        </div>
      </div>

      <div className="text-xs text-muted-foreground">{filtered.length} of {catalog.length} models</div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {filtered.map(m => (
          <div key={m.canonical_id} className="card border-border/50 space-y-2">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="font-semibold text-sm">{m.display_name}</span>
              <span className="badge badge-secondary text-[11px]">{m.family}</span>
              <span className="badge badge-secondary text-[11px]">{m.architecture.replace('_', ' ')}</span>
            </div>
            <p className="text-xs text-muted-foreground">{m.notes}</p>
            <div className="text-xs space-y-1">
              <div className="flex items-center gap-2 text-muted-foreground">
                <span className="font-mono opacity-70 w-12">SIZE</span>
                <span><strong className="text-foreground">{m.total_params_b}B</strong> params · {m.default_steps} default steps</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <span className="font-mono opacity-70 w-12">VRAM</span>
                <span>
                  FP16 <strong className="text-foreground">{m.vram_gb.fp16?.toFixed(1)}</strong> ·
                  Q8 <strong className="text-foreground">{m.vram_gb.q8?.toFixed(1)}</strong> ·
                  Q4 <strong className="text-foreground">{m.vram_gb.q4?.toFixed(1)}</strong> GB
                </span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <span className="font-mono opacity-70 w-12">DOES</span>
                <span>{m.supports.map(s => s.replace('image_', '')).join(' · ')}</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <span className="font-mono opacity-70 w-12">RUNS</span>
                <span className="text-[11px]">{m.harnesses_compatible.map(harnessName).join(', ')}</span>
              </div>
              <div className="flex items-center gap-2 text-muted-foreground">
                <span className="font-mono opacity-70 w-12">LIC</span>
                <span className="text-[11px]">{m.license}</span>
              </div>
            </div>
            {m.hf_id && (
              <a
                href={`https://huggingface.co/${m.hf_id}`}
                target="_blank" rel="noopener noreferrer"
                className="text-xs text-primary hover:underline inline-flex items-center gap-1"
              >
                <ExternalLink className="h-3 w-3" />
                {m.hf_id}
              </a>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}


function ImageRecCard({ rec, highlight, top }: {
  rec: ImageRecommendation; highlight?: boolean; top: ImageRecommendation
}) {
  const [expanded, setExpanded] = useState(false)
  const showComparison = rec.rank > 1 && top && top.canonical_id !== rec.canonical_id

  const confColor = rec.confidence_pct >= 80 ? 'bg-accent/15 text-accent'
    : rec.confidence_pct >= 50 ? 'bg-primary/15 text-primary' : 'bg-yellow-400/15 text-yellow-400'

  const timeStr = rec.estimated_time_per_image_s >= 60
    ? `${(rec.estimated_time_per_image_s / 60).toFixed(1)} min`
    : `${rec.estimated_time_per_image_s.toFixed(0)}s`

  return (
    <div className={cn('card border', highlight && scoreBg(rec.fit_score))}>
      <div className="flex items-start gap-4">
        <div className={cn('flex h-14 w-14 flex-col items-center justify-center rounded-xl shrink-0', scoreColor(rec.fit_score))}>
          <span className="text-xl font-bold leading-none">{rec.fit_score.toFixed(1)}</span>
          <span className="text-[10px] mt-0.5 opacity-70">fit</span>
        </div>
        <div className="flex-1 min-w-0 space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="badge badge-secondary text-[11px]">#{rec.rank}</span>
            {highlight && <Star className="h-4 w-4 text-yellow-400 fill-yellow-400" />}
            <span className="font-semibold">{rec.display_name}</span>
            <span className="badge badge-secondary text-[11px]">{rec.family}</span>
            <span className="badge badge-secondary text-[11px]">{rec.architecture.replace('_', ' ')}</span>
            <span className={cn('badge text-[11px]', confColor)} title={`${rec.benchmarks_measured} of ${rec.benchmarks_expected} benchmarks`}>
              {rec.confidence_pct}% confidence
            </span>
          </div>
          <p className="text-sm text-muted-foreground">{rec.why}</p>
          {showComparison && (
            <p className="text-xs text-muted-foreground italic">
              <span className="opacity-70">vs #1:</span> {buildImageComparison(rec, top)}
            </p>
          )}

          <div className="grid grid-cols-3 gap-3 pt-1">
            <ScoreBar label="Use case fit" score={rec.use_case_score} />
            <ScoreBar label="Hardware fit" score={rec.hardware_fit} />
            <ScoreBar label="Harness fit" score={rec.harness_fit} />
          </div>

          <div className="text-xs space-y-1 pt-1">
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="font-mono opacity-70 w-10">VRAM</span>
              <span>
                <strong className="text-foreground">{rec.estimated_vram_gb.toFixed(1)} GB</strong>
                <span className="opacity-70"> at {rec.quantization_recommended.toUpperCase()}</span>
              </span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="font-mono opacity-70 w-10">TIME</span>
              <span>
                <strong className="text-foreground">~{timeStr}</strong>
                <span className="opacity-70"> per 1024² image · {rec.default_steps} steps</span>
              </span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="font-mono opacity-70 w-10">LIC</span>
              <span className="text-[11px]">{rec.license}</span>
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
          {rec.notes && <p className="text-xs text-muted-foreground italic">{rec.notes}</p>}
          {rec.provenance.use_case_components.length > 0 && (
            <div className="space-y-1">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase">Benchmark evidence</h4>
              {rec.provenance.use_case_components.map(e => (
                <div key={e.benchmark} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">
                    {prettifyBenchmark(e.benchmark)} <span className="opacity-60">via {e.source}</span>
                  </span>
                  <span className="font-mono">
                    {formatBenchmarkValue(e.benchmark, e.value)} <span className="opacity-60">({(e.normalized * 100).toFixed(0)}%)</span>
                  </span>
                </div>
              ))}
            </div>
          )}
          {rec.install_options.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase">Install</h4>
              {rec.install_options.map((opt, i) => <ImageInstallRow key={i} option={opt} />)}
            </div>
          )}
        </div>
      )}
    </div>
  )
}


function ImageInstallRow({ option }: { option: ImageInstallOption }) {
  const [copied, setCopied] = useState(false)
  const handleCopy = async () => {
    try {
      await copyToClipboard(option.command)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch { /* clipboard unavailable */ }
  }
  return (
    <div className="flex items-start gap-2 text-xs">
      <span className="badge badge-secondary text-[10px] shrink-0 mt-0.5">{option.harness}</span>
      <code className="flex-1 font-mono text-[11px] bg-secondary/40 rounded px-2 py-1 break-all">{option.command}</code>
      <button onClick={handleCopy} className="btn-outline h-7 w-7 p-0 shrink-0" title="Copy">
        {copied ? <Check className="h-3 w-3 text-accent" /> : <Copy className="h-3 w-3" />}
      </button>
      {option.download_url && (
        <a href={option.download_url} target="_blank" rel="noopener noreferrer" className="btn-outline h-7 w-7 p-0 shrink-0" title="Open on Hugging Face">
          <ExternalLink className="h-3 w-3" />
        </a>
      )}
    </div>
  )
}


function buildImageComparison(rec: ImageRecommendation, top: ImageRecommendation): string {
  const parts: string[] = []
  const fitGap = top.fit_score - rec.fit_score
  if (fitGap > 0.05) parts.push(`-${fitGap.toFixed(1)} fit`)
  const tDelta = rec.estimated_time_per_image_s - top.estimated_time_per_image_s
  if (Math.abs(tDelta) > 1) {
    parts.push(tDelta > 0 ? `+${tDelta.toFixed(0)}s slower` : `${tDelta.toFixed(0)}s faster`)
  }
  const vramDelta = rec.estimated_vram_gb - top.estimated_vram_gb
  if (Math.abs(vramDelta) > 0.5) {
    parts.push(vramDelta > 0 ? `+${vramDelta.toFixed(1)} GB VRAM` : `${vramDelta.toFixed(1)} GB VRAM`)
  }
  return parts.length > 0 ? parts.join(', ') : 'similar profile'
}


function prettifyBenchmark(b: string): string {
  switch (b) {
    case 'geneval': return 'GenEval'
    case 'imagen_arena_elo': return 'imagen-arena ELO'
    case 'mjhq30k_fid': return 'MJHQ-30K FID'
    case 'emu_edit': return 'Emu Edit accuracy'
    default: return b
  }
}


function formatBenchmarkValue(benchmark: string, value: number): string {
  if (benchmark === 'imagen_arena_elo') return Math.round(value).toString()
  if (benchmark === 'mjhq30k_fid') return value.toFixed(1)
  return value.toFixed(2)
}


function ImageScoreExplainer({ onClose }: { onClose: () => void }) {
  return (
    <div className="bg-secondary/30 border border-border/60 rounded-md p-3 text-xs space-y-2">
      <div className="flex items-start justify-between">
        <h3 className="font-semibold">How image scores work</h3>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground">close</button>
      </div>
      <pre className="font-mono text-[11px] bg-background/60 rounded p-2 overflow-x-auto">
{`fit_score = 0.55 × use_case_score    (GenEval / arena ELO / Emu-Edit)
          + 0.30 × hardware_fit       (VRAM fit + time-per-image)
          + 0.15 × harness_fit        (Drawthings / ComfyUI / Mochi…)`}
      </pre>
      <div className="text-muted-foreground">
        Image gen is compute-bound: time-per-image scales with chip FP16 TFLOPS, not memory bandwidth.
        Reference times are measured on M3 Max / M4 Max and scaled by relative TFLOPS for other chips.
      </div>
    </div>
  )
}

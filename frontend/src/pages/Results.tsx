import { useEffect, useMemo, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { ArrowLeft, ChevronDown, ChevronUp, Loader2, Star, AlertTriangle, Eye, Info } from 'lucide-react'
import { useRecommend } from '../hooks/useRecommend'
import { useMeta } from '../hooks/useMeta'
import { ScoreBar } from '../components/ScoreBar'
import { InstallCommand } from '../components/InstallCommand'
import { cn, formatSize, scoreBg, scoreColor } from '../lib/utils'
import type { Recommendation } from '../types'

type SortKey = 'fit' | 'speed' | 'smallest' | 'benchmarks' | 'reasoning'
const SORT_OPTIONS: { value: SortKey; label: string; tooltip: string }[] = [
  { value: 'fit',        label: 'Best overall fit',  tooltip: 'Default — balanced across use case, hardware, and harness' },
  { value: 'benchmarks', label: 'Highest benchmarks', tooltip: 'Raw quality on the use-case benchmarks; ignores hardware' },
  { value: 'speed',      label: 'Fastest decode',     tooltip: 'Estimated tokens/sec on your hardware (highest first)' },
  { value: 'smallest',   label: 'Smallest footprint', tooltip: 'Lowest RAM use — good for keeping other apps open' },
  { value: 'reasoning',  label: 'Best reasoning',     tooltip: 'Sorted by GPQA + BBH (multi-step + graduate-level reasoning)' },
]

function applySort(recs: Recommendation[], sort: SortKey): Recommendation[] {
  const sorted = [...recs]
  switch (sort) {
    case 'fit':
      sorted.sort((a, b) => b.fit_score - a.fit_score); break
    case 'benchmarks':
      sorted.sort((a, b) => b.use_case_score - a.use_case_score); break
    case 'speed': {
      const tps = (r: Recommendation) => (r.estimated_tokens_per_sec[0] + r.estimated_tokens_per_sec[1]) / 2
      sorted.sort((a, b) => tps(b) - tps(a)); break
    }
    case 'smallest':
      sorted.sort((a, b) => (a.estimated_size_mb + a.estimated_kv_cache_mb) - (b.estimated_size_mb + b.estimated_kv_cache_mb)); break
    case 'reasoning': {
      const r = (rec: Recommendation) => {
        const gpqa = rec.provenance.use_case_components.find(e => e.benchmark === 'gpqa')?.normalized || 0
        const bbh  = rec.provenance.use_case_components.find(e => e.benchmark === 'bbh')?.normalized || 0
        return gpqa + bbh
      }
      sorted.sort((a, b) => r(b) - r(a)); break
    }
  }
  return sorted
}

export default function Results() {
  const [params] = useSearchParams()
  const { recommendations, hardware, loading, error, fetch } = useRecommend()
  const { useCases, harnesses } = useMeta()
  const [includeBig, setIncludeBig] = useState(false)
  const [sort, setSort] = useState<SortKey>('fit')
  const [showExplainer, setShowExplainer] = useState(false)

  const useCase = params.get('use_case') || ''
  const harness = params.get('harness') || null

  useEffect(() => {
    if (useCase) {
      fetch({ use_case: useCase, harness, limit: 15, include_too_big: includeBig })
    }
  }, [useCase, harness, includeBig, fetch])

  const useCaseLabel = useCases.find((u) => u.id === useCase)?.name || useCase
  const harnessLabel = harness ? harnesses.find((h) => h.id === harness)?.name : null

  const sortedRecs = useMemo(() => applySort(recommendations, sort), [recommendations, sort])

  if (loading && recommendations.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-20">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
        <p className="text-muted-foreground">Computing recommendations…</p>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back
      </Link>

      <header className="space-y-3">
        <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
          <span className="rounded-full bg-secondary px-2 py-0.5">{useCaseLabel}</span>
          {harnessLabel && <span className="rounded-full bg-secondary px-2 py-0.5">{harnessLabel}</span>}
        </div>
        <div className="flex items-baseline justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-2">
            <h1 className="text-2xl font-bold">Top recommendations</h1>
            <button
              onClick={() => setShowExplainer(!showExplainer)}
              className="inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-primary"
              aria-label="What do these scores mean?"
            >
              <Info className="h-3.5 w-3.5" />
              How is this scored?
            </button>
          </div>
          <div className="flex items-center gap-2">
            <label className="text-xs text-muted-foreground">Sort:</label>
            <select
              value={sort}
              onChange={(e) => setSort(e.target.value as SortKey)}
              className="input h-9 max-w-[200px] text-sm"
              title={SORT_OPTIONS.find(o => o.value === sort)?.tooltip}
            >
              {SORT_OPTIONS.map((o) => (
                <option key={o.value} value={o.value} title={o.tooltip}>{o.label}</option>
              ))}
            </select>
          </div>
        </div>

        {showExplainer && <ScoreExplainer onClose={() => setShowExplainer(false)} />}
      </header>

      {hardware && (
        <div className="card text-sm">
          <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-muted-foreground">
            <span><strong className="text-foreground">{hardware.chip}</strong></span>
            <span>{hardware.gpu_cores} GPU cores</span>
            <span>{hardware.total_memory_gb} GB RAM ({hardware.available_memory_gb.toFixed(1)} free)</span>
            <span>{hardware.neural_engine_cores}-core NPU · {hardware.generation}</span>
          </div>
        </div>
      )}

      {error && <div className="card border-destructive/30 bg-destructive/5 text-destructive">{error}</div>}

      <div className="flex items-center justify-between">
        <label className="flex items-center gap-2 text-sm text-muted-foreground cursor-pointer">
          <input
            type="checkbox"
            checked={includeBig}
            onChange={(e) => setIncludeBig(e.target.checked)}
            className="rounded border-border"
          />
          Include models that won't fit
        </label>
        <span className="text-xs text-muted-foreground">{recommendations.length} results</span>
      </div>

      {recommendations.length === 0 && !loading && (
        <div className="card text-center space-y-3 py-10">
          <p className="text-foreground font-medium">No models matched all filters.</p>
          <p className="text-sm text-muted-foreground max-w-md mx-auto">Most likely causes:</p>
          <ul className="text-sm text-muted-foreground max-w-md mx-auto space-y-1 list-disc list-inside text-left">
            {harness && <li>The <strong>{harnessLabel}</strong> harness excluded everything (probably context length or tool-calling) — try without it.</li>}
            <li>Models exceed your available RAM — toggle "include models that won't fit" above.</li>
            <li>Use case requires capabilities (vision, tool-calling, long context) that no model in the catalog has yet — refresh data sources.</li>
          </ul>
          <div className="flex gap-2 justify-center pt-2">
            {harness && (
              <Link
                to={`/results?use_case=${useCase}`}
                className="btn-outline text-sm"
              >
                Remove harness filter
              </Link>
            )}
            <Link to="/sources" className="btn-outline text-sm">Check data sources</Link>
          </div>
        </div>
      )}

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

      {recommendations.length >= 2 && (
        <div className="text-center">
          <Link
            to={`/compare?a=${recommendations[0].canonical_id}&b=${recommendations[1].canonical_id}`}
            className="btn-outline gap-2 inline-flex"
          >
            <Eye className="h-4 w-4" />
            Compare top 2 side-by-side
          </Link>
        </div>
      )}
    </div>
  )
}

function RecCard({ rec, position, highlight, sort, top, hardware }: {
  rec: Recommendation
  position: number
  highlight?: boolean
  sort: SortKey
  top: Recommendation
  hardware: { total_memory_gb: number; available_memory_gb: number; chip: string } | null
}) {
  const [expanded, setExpanded] = useState(false)

  // Headline number adapts to the active sort.
  const headlineValue = sort === 'speed'
    ? Math.round((rec.estimated_tokens_per_sec[0] + rec.estimated_tokens_per_sec[1]) / 2)
    : sort === 'smallest'
      ? Math.round((rec.estimated_size_mb + rec.estimated_kv_cache_mb) / 1024 * 10) / 10
      : sort === 'benchmarks'
        ? rec.use_case_score
        : rec.fit_score
  const headlineSuffix = sort === 'speed' ? 'tps' : sort === 'smallest' ? 'GB' : ''

  // Memory math — show the actual RAM situation
  const totalGb = hardware?.total_memory_gb ?? 0
  const residentGb = (rec.estimated_size_mb + rec.estimated_kv_cache_mb) / 1024
  const ramPct = totalGb > 0 ? (residentGb / totalGb) * 100 : 0

  // TPS math — bandwidth ÷ active params × efficiency
  const avgTps = (rec.estimated_tokens_per_sec[0] + rec.estimated_tokens_per_sec[1]) / 2

  // Comparison to #1 — only meaningful for non-top rows
  const showComparison = position > 1 && top && top.canonical_id !== rec.canonical_id
  const comparisonLine = showComparison ? buildComparison(rec, top) : null

  // Confidence chip color
  const confColor = rec.confidence_pct >= 80
    ? 'bg-accent/15 text-accent'
    : rec.confidence_pct >= 50
      ? 'bg-primary/15 text-primary'
      : 'bg-yellow-400/15 text-yellow-400'

  return (
    <div className={cn('card border', highlight && scoreBg(rec.fit_score))}>
      <div className="flex items-start gap-4">
        <div className={cn('flex h-14 w-14 flex-col items-center justify-center rounded-xl shrink-0',
                          scoreColor(rec.fit_score))}>
          <span className="text-xl font-bold leading-none">{Number(headlineValue).toFixed(headlineSuffix === '' ? 1 : 0)}</span>
          {headlineSuffix && <span className="text-[10px] mt-0.5 opacity-70">{headlineSuffix}</span>}
        </div>
        <div className="flex-1 min-w-0 space-y-2">
          {/* Header: name + tags */}
          <div className="flex items-center gap-2 flex-wrap">
            <span className="badge badge-secondary text-[11px]">#{position}</span>
            {highlight && sort === 'fit' && <Star className="h-4 w-4 text-yellow-400 fill-yellow-400" />}
            <Link to={`/model/${encodeURIComponent(rec.canonical_id)}`}
                  className="font-semibold hover:text-primary transition-colors">
              {rec.display_name}
            </Link>
            <span className="badge badge-secondary text-[11px]">{rec.parameter_size}</span>
            {rec.is_moe && <span className="badge badge-accent text-[11px]">MoE</span>}
            <span
              className={cn('badge text-[11px]', confColor)}
              title={`${rec.benchmarks_measured} of ${rec.benchmarks_expected} use-case benchmarks measured`}
            >
              {rec.confidence_pct}% confidence
            </span>
          </div>

          {/* Why summary */}
          <p className="text-sm text-muted-foreground">{rec.why}</p>

          {/* Comparison to #1 */}
          {comparisonLine && (
            <p className="text-xs text-muted-foreground italic">
              <span className="opacity-70">vs #1:</span> {comparisonLine}
            </p>
          )}

          {/* ALWAYS-VISIBLE BREAKDOWN — three sub-score bars */}
          <div className="grid grid-cols-3 gap-3 pt-2">
            <ScoreBar label="Use case fit" score={Math.min(10, rec.use_case_score)} />
            <ScoreBar label="Hardware fit" score={rec.hardware_fit} />
            <ScoreBar label="Harness fit" score={rec.harness_fit} />
          </div>

          {/* Memory math — visible by default */}
          <div className="text-xs space-y-1 pt-1">
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="font-mono opacity-70">RAM</span>
              <span>
                {formatSize(rec.estimated_size_mb)} weights
                {rec.estimated_kv_cache_mb > 0 && ` + ${formatSize(rec.estimated_kv_cache_mb)} KV`}
                {' = '}
                <strong className="text-foreground">{residentGb.toFixed(1)} GB</strong>
                {totalGb > 0 && (
                  <span className="opacity-70"> / {totalGb} GB total ({ramPct.toFixed(0)}%)</span>
                )}
              </span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="font-mono opacity-70">TPS</span>
              <span>
                {rec.bandwidth_gb_s} GB/s ÷ {rec.active_params_b}B {rec.is_moe && 'active'} × 0.70 ≈
                {' '}<strong className="text-foreground">{Math.round(avgTps)} tok/s decode</strong>
                <span className="opacity-70"> ({rec.estimated_tokens_per_sec[0]}–{rec.estimated_tokens_per_sec[1]} range)</span>
              </span>
            </div>
            <div className="flex items-center gap-2 text-muted-foreground">
              <span className="font-mono opacity-70">QNT</span>
              <span>
                <strong className="text-foreground">{rec.quantization_recommended}</strong>
                {rec.quant_quality_factor < 1.0 && (
                  <span className="opacity-70"> · ~{Math.round(rec.quant_quality_factor * 100)}% of FP16 quality</span>
                )}
              </span>
            </div>
          </div>

          {/* Warnings — yellow */}
          {rec.warnings.length > 0 && (
            <div className="flex items-start gap-1 text-xs text-yellow-400 pt-1">
              <AlertTriangle className="h-3 w-3 mt-0.5 shrink-0" />
              <span>{rec.warnings.join(' · ')}</span>
            </div>
          )}
        </div>

        <button
          onClick={() => setExpanded(!expanded)}
          className="btn-outline h-8 px-2 text-xs gap-1 shrink-0"
        >
          {expanded ? 'Less' : 'Details'}
          {expanded ? <ChevronUp className="h-3 w-3" /> : <ChevronDown className="h-3 w-3" />}
        </button>
      </div>

      {expanded && (
        <div className="mt-4 pt-4 border-t space-y-4">
          {rec.provenance.use_case_components.length > 0 && (
            <div className="space-y-1">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase">Benchmark evidence</h4>
              <div className="space-y-1">
                {rec.provenance.use_case_components.map((e) => (
                  <div key={e.benchmark} className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">
                      {e.benchmark} <span className="opacity-60">via {e.source}</span>
                    </span>
                    <span className="font-mono">
                      {e.value.toFixed(1)} <span className="opacity-60">({(e.normalized * 100).toFixed(0)}%)</span>
                    </span>
                  </div>
                ))}
              </div>
              {rec.benchmarks_measured < rec.benchmarks_expected && (
                <p className="text-xs text-muted-foreground italic pt-1">
                  Missing: {rec.provenance.missing_data
                    .filter(m => m.startsWith('use_case:'))
                    .map(m => m.slice('use_case:'.length))
                    .join(', ') || '(none)'}
                </p>
              )}
            </div>
          )}

          {rec.install_options.length > 0 && (
            <div className="space-y-2">
              <h4 className="text-xs font-semibold text-muted-foreground uppercase">Install</h4>
              <div className="space-y-2">
                {rec.install_options.map((opt, i) => (
                  <InstallCommand key={i} option={opt} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function buildComparison(rec: Recommendation, top: Recommendation): string {
  const parts: string[] = []
  // Score gap
  const fitGap = top.fit_score - rec.fit_score
  if (fitGap > 0.05) parts.push(`-${fitGap.toFixed(1)} fit`)

  // Speed comparison
  const recTps = (rec.estimated_tokens_per_sec[0] + rec.estimated_tokens_per_sec[1]) / 2
  const topTps = (top.estimated_tokens_per_sec[0] + top.estimated_tokens_per_sec[1]) / 2
  const tpsRatio = recTps / Math.max(topTps, 1)
  if (tpsRatio > 1.3) parts.push(`${Math.round(recTps)} vs ${Math.round(topTps)} tok/s (faster)`)
  else if (tpsRatio < 0.77) parts.push(`${Math.round(recTps)} vs ${Math.round(topTps)} tok/s (slower)`)

  // Size comparison
  const recGb = (rec.estimated_size_mb + rec.estimated_kv_cache_mb) / 1024
  const topGb = (top.estimated_size_mb + top.estimated_kv_cache_mb) / 1024
  const sizeDelta = recGb - topGb
  if (Math.abs(sizeDelta) > 1) parts.push(sizeDelta > 0 ? `+${sizeDelta.toFixed(1)} GB RAM` : `${sizeDelta.toFixed(1)} GB RAM`)

  // Coverage
  if (rec.confidence_pct < top.confidence_pct - 10) {
    parts.push(`thinner data (${rec.confidence_pct}% vs ${top.confidence_pct}%)`)
  }

  return parts.length > 0 ? parts.join(', ') : 'similar profile'
}

function ScoreExplainer({ onClose }: { onClose: () => void }) {
  return (
    <div className="card bg-secondary/30 border-border/60 text-sm space-y-3">
      <div className="flex items-start justify-between gap-3">
        <h3 className="font-semibold">How the scores work</h3>
        <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-xs">close</button>
      </div>
      <p className="text-muted-foreground text-xs leading-relaxed">
        The big number on each card is the <strong className="text-foreground">fit score</strong> for the active sort.
        For "Best overall fit" (default) it combines three components:
      </p>
      <pre className="font-mono text-xs bg-background/60 rounded p-3 overflow-x-auto">
{`fit_score = 0.55 × use_case_score    (benchmark match for the task)
          + 0.30 × hardware_fit       (RAM fit + decode speed)
          + 0.15 × harness_fit        (compatibility with chosen agent)`}
      </pre>
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-1.5 text-xs">
        <div><span className="text-accent font-mono">9.0+</span> <span className="text-muted-foreground">excellent — top of class for your hardware</span></div>
        <div><span className="text-primary font-mono">7.0–9.0</span> <span className="text-muted-foreground">solid pick</span></div>
        <div><span className="text-yellow-400 font-mono">5.0–7.0</span> <span className="text-muted-foreground">workable but compromised</span></div>
        <div><span className="text-destructive font-mono">&lt; 5.0</span> <span className="text-muted-foreground">missing benchmarks or poor fit</span></div>
      </div>
      <p className="text-xs text-muted-foreground">
        Sub-scores are 0–10. Use-case score can exceed 10 when a chosen harness boosts your task
        (e.g. Cline boosts coding × 1.5), so total fit can read up to ~12 for an ideal match.
        Click <strong className="text-foreground">"Why?"</strong> on any row to see which benchmarks contributed.
      </p>
    </div>
  )
}

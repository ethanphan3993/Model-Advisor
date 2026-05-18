import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { Brain, Cpu, Zap, HardDrive, ArrowRight, Loader2, Database, AlertCircle, RefreshCw } from 'lucide-react'
import { useScan } from '../hooks/useScan'
import { useMeta } from '../hooks/useMeta'
import { listModels, triggerRefresh } from '../lib/api'
import { cn, formatScore, scoreColor } from '../lib/utils'

export default function Home() {
  const { data, loading, scan } = useScan()
  const { useCases } = useMeta()

  // First-run / staleness detection — non-blocking probe of the catalog state.
  const [catalogState, setCatalogState] = useState<'checking' | 'empty' | 'stale' | 'fresh'>('checking')
  const [catalogTotal, setCatalogTotal] = useState(0)
  const [refreshing, setRefreshing] = useState(false)
  const [refreshError, setRefreshError] = useState<string | null>(null)

  useEffect(() => {
    if (!data && !loading) scan()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    listModels({ has_benchmarks: true, limit: 1 })
      .then(r => {
        setCatalogTotal(r.total)
        setCatalogState(r.total === 0 ? 'empty' : r.total < 100 ? 'stale' : 'fresh')
      })
      .catch(() => setCatalogState('empty'))
  }, [])

  const handleRefresh = async () => {
    setRefreshing(true)
    setRefreshError(null)
    try {
      await triggerRefresh()
      const r = await listModels({ has_benchmarks: true, limit: 1 })
      setCatalogTotal(r.total)
      setCatalogState(r.total === 0 ? 'empty' : r.total < 100 ? 'stale' : 'fresh')
    } catch (e) {
      setRefreshError(e instanceof Error ? e.message : 'Refresh failed')
    } finally {
      setRefreshing(false)
    }
  }

  return (
    <div className="space-y-8">
      {/* Hero */}
      <div className="text-center space-y-5 py-6">
        <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-2xl bg-primary/10">
          <Brain className="h-10 w-10 text-primary" />
        </div>
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Model Advisor</h1>
          <p className="text-base text-muted-foreground max-w-2xl mx-auto">
            What do you want to do? Pick a use case and we'll rank the best local models for your Mac
            across Ollama, LM Studio, and HuggingFace — hardware-aware (MoE-savvy, bandwidth-bound),
            with full benchmark provenance.
          </p>
        </div>
      </div>

      {/* First-run / empty-cache banner */}
      {(catalogState === 'empty' || catalogState === 'stale') && (
        <div className={cn(
          "card border flex items-start gap-3",
          catalogState === 'empty' ? "border-yellow-400/40 bg-yellow-400/5" : "border-primary/30 bg-primary/5"
        )}>
          {catalogState === 'empty' ? (
            <AlertCircle className="h-5 w-5 text-yellow-400 shrink-0 mt-0.5" />
          ) : (
            <Database className="h-5 w-5 text-primary shrink-0 mt-0.5" />
          )}
          <div className="flex-1 space-y-2">
            <div>
              <h3 className="font-semibold">
                {catalogState === 'empty'
                  ? 'No catalog data yet'
                  : `Catalog looks light (${catalogTotal} models with benchmarks)`}
              </h3>
              <p className="text-sm text-muted-foreground">
                {catalogState === 'empty'
                  ? 'Recommendations need benchmark data from the public sources. Run a refresh — it takes ~25 seconds.'
                  : 'Some sources may not have run yet. Refreshing pulls the latest from all sources.'}
              </p>
            </div>
            {refreshError && (
              <p className="text-xs text-destructive">{refreshError}</p>
            )}
            <div className="flex items-center gap-3">
              <button
                onClick={handleRefresh}
                disabled={refreshing}
                className="btn-primary gap-2"
              >
                {refreshing ? (
                  <><Loader2 className="h-4 w-4 animate-spin" /> Refreshing all sources…</>
                ) : (
                  <><RefreshCw className="h-4 w-4" /> Populate catalog</>
                )}
              </button>
              <Link to="/sources" className="text-xs text-muted-foreground hover:text-foreground">
                View source status →
              </Link>
            </div>
          </div>
        </div>
      )}

      {/* Hardware tile */}
      <div className="card border-border/50">
        {loading && (
          <div className="flex items-center gap-3 text-muted-foreground">
            <Loader2 className="h-5 w-5 animate-spin" />
            Scanning hardware…
          </div>
        )}
        {data && (
          <div className="flex items-start justify-between gap-6">
            <div className="space-y-1">
              <div className="text-sm text-muted-foreground">Detected device</div>
              <h2 className="text-xl font-semibold">{data.hardware.model}</h2>
              <p className="text-sm text-muted-foreground">{data.ai_capability.interpretation}</p>
            </div>
            <div className={cn('text-right', scoreColor(data.ai_capability.composite_score))}>
              <div className="text-3xl font-bold">{formatScore(data.ai_capability.composite_score)}</div>
              <div className="text-xs text-muted-foreground">capability</div>
            </div>
          </div>
        )}
        {data && (
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 mt-5">
            <Stat icon={Cpu} top={data.hardware.chip.chip} bottom={`${data.hardware.chip.gpu_cores} GPU cores`} />
            <Stat icon={Zap} top={`${data.memory.total_gb} GB RAM`} bottom={`${data.memory.available_gb.toFixed(1)} GB available`} />
            <Stat icon={HardDrive} top={`${data.hardware.chip.neural_engine_cores}-core NPU`} bottom={data.hardware.chip.generation} />
          </div>
        )}
      </div>

      {/* Use case picker */}
      <div className="space-y-3">
        <h2 className="text-lg font-semibold">What do you want to do?</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
          {useCases.map((uc) => (
            <Link
              key={uc.id}
              to={`/wizard/${uc.id}`}
              className="card-hover group"
            >
              <div className="flex items-start justify-between gap-3">
                <div className="space-y-1">
                  <h3 className="font-semibold group-hover:text-primary transition-colors">{uc.name}</h3>
                  <p className="text-xs text-muted-foreground">{uc.tagline}</p>
                </div>
                <ArrowRight className="h-4 w-4 text-muted-foreground group-hover:text-primary transition-colors" />
              </div>
            </Link>
          ))}
        </div>
      </div>
    </div>
  )
}

function Stat({ icon: Icon, top, bottom }: { icon: React.ElementType; top: string; bottom: string }) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-secondary/40">
      <Icon className="h-5 w-5 text-primary shrink-0" />
      <div className="min-w-0">
        <div className="text-sm font-medium truncate">{top}</div>
        <div className="text-xs text-muted-foreground truncate">{bottom}</div>
      </div>
    </div>
  )
}

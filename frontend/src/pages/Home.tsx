import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Brain, Cpu, Zap, HardDrive, ArrowRight, Loader2 } from 'lucide-react'
import { useScan } from '../hooks/useScan'
import { useMeta } from '../hooks/useMeta'
import { cn, formatScore, scoreColor } from '../lib/utils'

export default function Home() {
  const { data, loading, scan } = useScan()
  const { useCases } = useMeta()

  useEffect(() => {
    if (!data && !loading) scan()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  return (
    <div className="space-y-10">
      {/* Hero */}
      <div className="text-center space-y-5 py-6">
        <div className="mx-auto flex h-20 w-20 items-center justify-center rounded-2xl bg-primary/10">
          <Brain className="h-10 w-10 text-primary" />
        </div>
        <div className="space-y-2">
          <h1 className="text-4xl font-bold tracking-tight">Model Advisor</h1>
          <p className="text-base text-muted-foreground max-w-2xl mx-auto">
            What do you want to do? Pick a use case and we'll rank the best local models for your Mac
            across Ollama, LM Studio, and HuggingFace — with persona match, hardware fit, and benchmark provenance.
          </p>
        </div>
      </div>

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

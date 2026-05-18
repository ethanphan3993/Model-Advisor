import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { Cpu, Zap, HardDrive, Monitor, Battery, Server } from 'lucide-react'
import { useScan } from '../hooks/useScan'
import { cn, formatScore, scoreColor, scoreBg } from '../lib/utils'

export default function ScanResults() {
  const { data, loading, error, scan } = useScan()

  useEffect(() => {
    if (!data) scan()
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="skeleton h-8 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => <div key={i} className="skeleton h-48" />)}
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="card border-destructive/30 bg-destructive/5">
        <h2 className="text-lg font-semibold text-destructive mb-2">Scan Failed</h2>
        <p className="text-muted-foreground mb-4">{error}</p>
        <p className="text-sm text-muted-foreground">
          Hardware scanning requires macOS. Make sure you're running this app on a Mac.
        </p>
        <button onClick={() => scan()} className="btn-primary mt-4">
          Retry Scan
        </button>
      </div>
    )
  }

  if (!data) return null

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Scan Results</h1>
        <button onClick={() => scan()} className="btn-outline">
          Refresh
        </button>
      </div>

      {/* Device Header */}
      <div className="card">
        <div className="flex items-start justify-between">
          <div>
            <h2 className="text-xl font-semibold">{data.hardware.model}</h2>
            <p className="text-muted-foreground">
              {data.hardware.model_identifier} · {data.hardware.chip.generation}
            </p>
          </div>
          <div className={cn('text-3xl font-bold', scoreColor(data.ai_capability.composite_score))}>
            {formatScore(data.ai_capability.composite_score)}
          </div>
        </div>
      </div>

      {/* AI Capability Scores */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">AI Capability Scores</h3>
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <ScoreItem label="GPU Score" score={data.ai_capability.gpu_score} detail={`${data.gpu.gpu_cores} cores`} />
          <ScoreItem label="Memory Score" score={data.ai_capability.memory_score} detail={`${data.memory.total_gb} GB unified`} />
          <ScoreItem label="Neural Engine" score={data.ai_capability.neural_engine_score} detail={`${data.hardware.chip.neural_engine_cores} cores`} />
        </div>
        <p className="text-sm text-muted-foreground mt-4">{data.ai_capability.interpretation}</p>
      </div>

      {/* Hardware Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <HardwareCard
          icon={Cpu}
          title="Chip & CPU"
          details={[
            `${data.hardware.chip.chip}`,
            `${data.hardware.chip.cpu_cores.total} cores (${data.hardware.chip.cpu_cores.super}P + ${data.hardware.chip.cpu_cores.performance}E)`,
            `${data.hardware.chip.gpu_cores} GPU cores`,
            `${data.cpu_extra.logical_cores} logical / ${data.cpu_extra.physical_cores} physical`,
          ]}
        />
        <HardwareCard
          icon={Zap}
          title="Memory"
          details={[
            `${data.memory.total_gb} GB total`,
            `${data.memory.available_gb.toFixed(1)} GB available (${((data.memory.available_gb / data.memory.total_gb) * 100).toFixed(0)}% free)`,
            `${data.memory.used_gb.toFixed(1)} GB used`,
            `${data.memory.compressed_gb.toFixed(1)} GB compressed`,
          ]}
        />
        <HardwareCard
          icon={HardDrive}
          title="Storage"
          details={data.storage.map((s) => `${s.name}: ${s.free_gb.toFixed(1)} / ${s.capacity_gb.toFixed(1)} GB (${s.media_type})`)}
        />
        <HardwareCard
          icon={Monitor}
          title="Display"
          details={[
            data.display.model || 'Built-in',
            data.display.resolution || 'N/A',
            data.display.main ? 'Main display' : '',
          ].filter(Boolean)}
        />
        <HardwareCard
          icon={Battery}
          title="Battery"
          details={data.power ? [
            data.power.battery_health || 'N/A',
            `${data.power.cycle_count} cycles`,
            `${data.power.charge_pct}% charged`,
            data.power.is_charging ? 'Charging' : 'On battery',
          ] : ['Not applicable (desktop)']}
        />
        <HardwareCard
          icon={Server}
          title="Operating System"
          details={[
            `${data.os.name} ${data.os.version}`,
            `Build ${data.os.build}`,
            `Boot ROM ${data.hardware.boot_rom_version || 'N/A'}`,
          ]}
        />
      </div>

      <div className="text-center">
        <Link to="/" className="btn-primary">
          Get Model Recommendations
        </Link>
      </div>
    </div>
  )
}

function ScoreItem({ label, score, detail }: { label: string; score: number; detail: string }) {
  return (
    <div className={cn('card border', scoreBg(score))}>
      <div className="text-sm text-muted-foreground">{label}</div>
      <div className={cn('text-2xl font-bold', scoreColor(score))}>{score}/10</div>
      <div className="text-xs text-muted-foreground mt-1">{detail}</div>
    </div>
  )
}

function HardwareCard({ icon: Icon, title, details }: { icon: React.ElementType; title: string; details: string[] }) {
  return (
    <div className="card">
      <div className="flex items-center gap-2 mb-3">
        <Icon className="h-5 w-5 text-primary" />
        <h3 className="font-semibold">{title}</h3>
      </div>
      <div className="space-y-1">
        {details.map((d, i) => (
          <p key={i} className="text-sm text-muted-foreground">{d}</p>
        ))}
      </div>
    </div>
  )
}

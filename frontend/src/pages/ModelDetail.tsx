import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ArrowLeft } from 'lucide-react'
import { getModel } from '../lib/api'
import { ScoreBar } from '../components/ScoreBar'
import { InstallCommand } from '../components/InstallCommand'
import type { ModelDetail } from '../types'

export default function ModelDetailPage() {
  const { canonicalId } = useParams<{ canonicalId: string }>()
  const [model, setModel] = useState<ModelDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!canonicalId) return
    setLoading(true)
    getModel(canonicalId)
      .then(setModel)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load model'))
      .finally(() => setLoading(false))
  }, [canonicalId])

  if (loading) {
    return (
      <div className="space-y-4">
        <div className="skeleton h-8 w-64" />
        <div className="skeleton h-64" />
      </div>
    )
  }

  if (error || !model) {
    return (
      <div className="card border-destructive/30 bg-destructive/5">
        <p className="text-destructive">{error || 'Not found'}</p>
        <Link to="/browse" className="btn-outline mt-4 inline-block">Back to Browse</Link>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Link to="/browse" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back to Browse
      </Link>

      <div className="card">
        <div className="space-y-2">
          <div className="flex items-center gap-2 flex-wrap">
            <h1 className="text-2xl font-bold">{model.display_name}</h1>
            <span className="badge badge-secondary">{model.parameter_size}</span>
            <span className="badge badge-secondary">{model.variant}</span>
            {model.vision && <span className="badge badge-accent">vision</span>}
            {model.tool_calling && <span className="badge badge-primary">tool-calling</span>}
          </div>
          <p className="text-sm text-muted-foreground">
            Family: <strong>{model.family}</strong>
            {model.license && <> · License: <strong>{model.license}</strong></>}
            {(model.context_length || 0) > 0 && <> · Context: <strong>{(model.context_length / 1000).toFixed(0)}K</strong></>}
          </p>
          {model.description && (
            <p className="text-sm text-muted-foreground pt-2">{model.description}</p>
          )}
        </div>
      </div>

      {/* Benchmarks */}
      {model.scores.length > 0 && (
        <div className="card space-y-3">
          <h2 className="font-semibold">Benchmark scores</h2>
          <p className="text-xs text-muted-foreground">{model.scores.length} measurements across all sources</p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {model.scores.map((s, i) => (
              <div key={i} className="space-y-1">
                <ScoreBar label={s.benchmark} score={s.normalized * 10} />
                <div className="text-xs text-muted-foreground">
                  raw: <span className="font-mono">{s.value.toFixed(1)}</span> via <span className="font-mono">{s.source}</span> · {s.confidence}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Install */}
      {model.artifacts.length > 0 && (
        <div className="card space-y-3">
          <h2 className="font-semibold">How to install</h2>
          <div className="space-y-2">
            {model.artifacts.map((a, i) => (
              <InstallCommand key={i} option={{
                harness: null,
                source: a.source,
                source_id: a.source_id,
                command: a.install_command,
                url: a.download_url,
                size_mb: a.size_mb,
                quantization: a.quantization,
              }} />
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

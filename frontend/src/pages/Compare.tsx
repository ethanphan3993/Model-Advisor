import { useEffect, useState } from 'react'
import { Link, useSearchParams } from 'react-router-dom'
import { ArrowLeft, Loader2 } from 'lucide-react'
import { getModel } from '../lib/api'
import { ScoreBar } from '../components/ScoreBar'
import type { ModelDetail } from '../types'

export default function Compare() {
  const [params] = useSearchParams()
  const a = params.get('a') || ''
  const b = params.get('b') || ''
  const [modelA, setA] = useState<ModelDetail | null>(null)
  const [modelB, setB] = useState<ModelDetail | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([getModel(a).catch(() => null), getModel(b).catch(() => null)])
      .then(([ra, rb]) => { setA(ra); setB(rb) })
      .finally(() => setLoading(false))
  }, [a, b])

  if (loading) {
    return <div className="flex justify-center py-20"><Loader2 className="h-8 w-8 animate-spin" /></div>
  }
  if (!modelA || !modelB) {
    return <div className="card border-destructive/30 bg-destructive/5">
      <p>Couldn't load one or both models.</p>
      <Link to="/" className="btn-outline mt-3 inline-block">Back home</Link>
    </div>
  }

  // Build the union of benchmarks
  const benchmarks = Array.from(new Set([
    ...modelA.scores.map(s => s.benchmark),
    ...modelB.scores.map(s => s.benchmark),
  ])).sort()

  return (
    <div className="space-y-6">
      <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back
      </Link>

      <h1 className="text-2xl font-bold">Compare</h1>

      <div className="grid grid-cols-2 gap-4">
        <Header model={modelA} />
        <Header model={modelB} />
      </div>

      <div className="card space-y-3">
        <h2 className="font-semibold">Benchmarks</h2>
        <div className="space-y-3">
          {benchmarks.map((bm) => {
            const sa = modelA.scores.find(s => s.benchmark === bm)
            const sb = modelB.scores.find(s => s.benchmark === bm)
            return (
              <div key={bm} className="grid grid-cols-2 gap-4 items-center">
                <div>
                  {sa ? <ScoreBar label={bm} score={sa.normalized * 10} /> :
                    <div className="text-xs text-muted-foreground">— no data for {bm}</div>}
                </div>
                <div>
                  {sb ? <ScoreBar label={bm} score={sb.normalized * 10} /> :
                    <div className="text-xs text-muted-foreground">— no data for {bm}</div>}
                </div>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

function Header({ model }: { model: ModelDetail }) {
  return (
    <div className="card">
      <h2 className="font-semibold">{model.display_name}</h2>
      <p className="text-xs text-muted-foreground">{model.family} · {model.parameter_size} · {model.variant}</p>
      <div className="text-xs text-muted-foreground mt-2">
        {model.scores.length} benchmarks · {model.artifacts.length} install options
      </div>
    </div>
  )
}

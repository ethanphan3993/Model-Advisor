import { useEffect, useState } from 'react'
import { RefreshCw, CheckCircle2, AlertCircle, Loader2 } from 'lucide-react'
import { getSources, triggerRefresh } from '../lib/api'
import { cn, formatTimeAgo } from '../lib/utils'
import type { SourceStatus } from '../types'

export default function Sources() {
  const [sources, setSources] = useState<SourceStatus[]>([])
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState<string | 'all' | null>(null)

  const reload = () => getSources().then((r) => setSources(r.sources)).finally(() => setLoading(false))

  useEffect(() => { reload() }, [])

  const handleRefresh = async (source?: string) => {
    setRefreshing(source ?? 'all')
    try {
      await triggerRefresh(source)
      await reload()
    } finally {
      setRefreshing(null)
    }
  }

  if (loading) return <div className="flex justify-center py-20"><Loader2 className="h-8 w-8 animate-spin" /></div>

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Data sources</h1>
        <button onClick={() => handleRefresh()} disabled={!!refreshing} className="btn-primary gap-2">
          <RefreshCw className={cn('h-4 w-4', refreshing === 'all' && 'animate-spin')} />
          {refreshing === 'all' ? 'Refreshing all…' : 'Refresh all'}
        </button>
      </div>

      <p className="text-sm text-muted-foreground">
        Public benchmark and catalog sources are merged into the local SQLite cache.
        Recommendations always read from the cache; these jobs keep it fresh.
        See <a href="https://github.com/ethanphan3993/Model-Advisor/blob/main/README.md#data-sources"
        target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">README → Data sources</a> for what each one provides.
      </p>

      <div className="space-y-2">
        {sources.length === 0 && (
          <div className="card text-center text-muted-foreground">
            No source has run yet. Click "Refresh all" to populate the cache.
          </div>
        )}
        {sources.map((s) => (
          <div key={s.source} className="card flex items-center justify-between">
            <div className="flex items-center gap-3 min-w-0">
              {s.last_status === 'ok' ? <CheckCircle2 className="h-5 w-5 text-accent shrink-0" /> :
                s.last_status === 'partial' ? <AlertCircle className="h-5 w-5 text-yellow-400 shrink-0" /> :
                <AlertCircle className="h-5 w-5 text-destructive shrink-0" />}
              <div className="min-w-0">
                <div className="font-semibold">{s.source}</div>
                <div className="text-xs text-muted-foreground truncate">
                  {s.last_status} · {s.rows_written} rows · {s.duration_ms}ms · {formatTimeAgo(s.last_run_at)}
                  {s.error_message && ` · ${s.error_message}`}
                </div>
              </div>
            </div>
            <button
              onClick={() => handleRefresh(s.source)}
              disabled={!!refreshing}
              className="btn-outline gap-2 shrink-0"
            >
              <RefreshCw className={cn('h-3.5 w-3.5', refreshing === s.source && 'animate-spin')} />
              Refresh
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}

import { useMemo, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { ArrowLeft, ArrowRight, Sparkles, Loader2, ExternalLink } from 'lucide-react'
import { useMeta } from '../hooks/useMeta'
import { cn } from '../lib/utils'
import type { Harness } from '../types'

// Display order — coding agents first, then general agents, MCP, chat UIs, runtimes.
const CATEGORY_ORDER = ['coding-agent', 'general-agent', 'tool-use', 'chat-ui', 'runtime'] as const
const CATEGORY_LABELS: Record<string, string> = {
  'coding-agent': 'Coding agents',
  'general-agent': 'General agents',
  'tool-use': 'MCP / tool clients',
  'chat-ui': 'Chat UIs',
  'runtime': 'Runtimes',
}

export default function Wizard() {
  const { useCaseId } = useParams<{ useCaseId: string }>()
  const navigate = useNavigate()
  const { useCases, harnesses, loading } = useMeta()
  const [harness, setHarness] = useState<string | null>(null)

  const grouped = useMemo(() => {
    const m = new Map<string, Harness[]>()
    for (const h of harnesses) {
      if (!m.has(h.category)) m.set(h.category, [])
      m.get(h.category)!.push(h)
    }
    return CATEGORY_ORDER
      .map((c) => ({ category: c, label: CATEGORY_LABELS[c], items: m.get(c) || [] }))
      .filter((g) => g.items.length > 0)
  }, [harnesses])

  if (loading) {
    return <div className="flex justify-center py-20"><Loader2 className="h-8 w-8 animate-spin text-muted-foreground" /></div>
  }

  const currentUseCase = useCases.find((u) => u.id === useCaseId)
  if (!currentUseCase) {
    return (
      <div className="card">
        <p>Unknown use case.</p>
        <Link to="/" className="btn-outline mt-3 inline-block">Back home</Link>
      </div>
    )
  }

  const submit = (chosen: string | null) => {
    const qs = new URLSearchParams()
    qs.set('use_case', useCaseId!)
    if (chosen) qs.set('harness', chosen)
    navigate(`/results?${qs.toString()}`)
  }

  return (
    <div className="space-y-8 max-w-3xl mx-auto">
      <Link to="/" className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground">
        <ArrowLeft className="h-4 w-4" /> Back
      </Link>

      <header className="space-y-1">
        <span className="text-xs uppercase tracking-wide text-primary">Step 2 of 2</span>
        <h1 className="text-2xl font-bold">Pick your agent harness</h1>
        <p className="text-muted-foreground">
          What's actually running the model? We'll filter for context length, tool-calling, and runtime
          compatibility, then rerank with the harness's strengths in mind. Skip if you'll wire it up yourself.
        </p>
      </header>

      <div className="space-y-6">
        {grouped.map((group) => (
          <section key={group.category} className="space-y-2">
            <h2 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">{group.label}</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {group.items.map((h) => (
                <button
                  key={h.id}
                  onClick={() => setHarness(harness === h.id ? null : h.id)}
                  className={cn(
                    'rounded-lg border p-3 text-left transition-colors',
                    harness === h.id
                      ? 'border-primary bg-primary/10'
                      : 'border-border hover:bg-secondary',
                  )}
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold">{h.name}</span>
                    {h.homepage && (
                      <a
                        href={h.homepage}
                        target="_blank"
                        rel="noopener noreferrer"
                        onClick={(e) => e.stopPropagation()}
                        className="text-muted-foreground hover:text-foreground"
                      >
                        <ExternalLink className="h-3 w-3" />
                      </a>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1">{h.description}</p>
                </button>
              ))}
            </div>
          </section>
        ))}
      </div>

      <div className="flex flex-col sm:flex-row sm:justify-between gap-3 sticky bottom-4">
        <button onClick={() => submit(null)} className="btn-outline gap-2">
          Skip — show all
        </button>
        <button onClick={() => submit(harness)} className="btn-primary gap-2" disabled={!harness}>
          <Sparkles className="h-4 w-4" />
          Show recommendations
          <ArrowRight className="h-4 w-4" />
        </button>
      </div>
    </div>
  )
}

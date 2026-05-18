import { cn, scoreColor } from '../lib/utils'

export function ScoreBar({ label, score, max = 10, className }: { label: string; score: number; max?: number; className?: string }) {
  const pct = Math.min(100, Math.max(0, (score / max) * 100))
  return (
    <div className={cn('space-y-1', className)}>
      <div className="flex items-center justify-between text-xs">
        <span className="text-muted-foreground">{label}</span>
        <span className={cn('font-mono font-semibold', scoreColor(score))}>{score.toFixed(1)}</span>
      </div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-secondary">
        <div
          className={cn(
            'h-full rounded-full transition-all',
            score >= 8 ? 'bg-accent' : score >= 5 ? 'bg-primary' : score >= 3 ? 'bg-yellow-400' : 'bg-destructive',
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}

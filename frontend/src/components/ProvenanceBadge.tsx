import { cn, confidenceBadgeClass } from '../lib/utils'

export function ProvenanceBadge({ confidence }: { confidence: 'high' | 'medium' | 'low' }) {
  return (
    <span className={cn('badge', confidenceBadgeClass(confidence))} title={`Recommendation confidence: ${confidence}`}>
      {confidence} confidence
    </span>
  )
}

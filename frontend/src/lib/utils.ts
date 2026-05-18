export function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ')
}

export function formatSize(mb: number): string {
  if (!mb || mb <= 0) return '—'
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB`
  return `${Math.round(mb)} MB`
}

export function formatScore(score: number, max = 10): string {
  return `${score.toFixed(1)}/${max}`
}

export function scoreColor(score: number): string {
  if (score >= 8) return 'text-accent'
  if (score >= 5) return 'text-primary'
  if (score >= 3) return 'text-yellow-400'
  return 'text-destructive'
}

export function scoreBg(score: number): string {
  if (score >= 8) return 'bg-accent/10 border-accent/30'
  if (score >= 5) return 'bg-primary/10 border-primary/30'
  if (score >= 3) return 'bg-yellow-400/10 border-yellow-400/30'
  return 'bg-destructive/10 border-destructive/30'
}

export function formatTPS(min: number, max: number): string {
  return `${min}–${max} tok/s`
}

export function formatTimeAgo(epochSec: number): string {
  if (!epochSec) return 'never'
  const diff = Math.floor(Date.now() / 1000) - epochSec
  if (diff < 60) return `${diff}s ago`
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

export function copyToClipboard(text: string): Promise<void> {
  if (navigator.clipboard) return navigator.clipboard.writeText(text)
  return Promise.reject(new Error('clipboard unavailable'))
}

export function confidenceBadgeClass(c: 'high' | 'medium' | 'low'): string {
  if (c === 'high') return 'bg-accent/15 text-accent'
  if (c === 'medium') return 'bg-primary/15 text-primary'
  return 'bg-yellow-400/15 text-yellow-400'
}

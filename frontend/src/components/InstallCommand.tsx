import { useState } from 'react'
import { Copy, Check, ExternalLink } from 'lucide-react'
import type { InstallOption } from '../types'
import { cn, copyToClipboard, formatSize } from '../lib/utils'

export function InstallCommand({ option }: { option: InstallOption }) {
  const [copied, setCopied] = useState(false)

  const handleCopy = async () => {
    if (!option.command) return
    await copyToClipboard(option.command)
    setCopied(true)
    setTimeout(() => setCopied(false), 1500)
  }

  return (
    <div className="rounded-md border bg-secondary/30 p-3 space-y-2">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={cn('badge', sourceColor(option.source))}>{sourceLabel(option.source)}</span>
          {option.quantization && (
            <span className="badge badge-secondary">{option.quantization}</span>
          )}
          {option.size_mb > 0 && (
            <span className="text-xs text-muted-foreground">{formatSize(option.size_mb)}</span>
          )}
        </div>
        {option.url && (
          <a href={option.url} target="_blank" rel="noopener noreferrer"
             className="text-muted-foreground hover:text-foreground">
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        )}
      </div>
      {option.command && (
        <div className="flex items-center gap-2">
          <code className="flex-1 truncate rounded bg-background px-2 py-1.5 font-mono text-xs">
            {option.command}
          </code>
          <button onClick={handleCopy} className="btn-outline h-8 px-2 text-xs">
            {copied ? <Check className="h-3.5 w-3.5 text-accent" /> : <Copy className="h-3.5 w-3.5" />}
          </button>
        </div>
      )}
    </div>
  )
}

function sourceLabel(source: string): string {
  if (source === 'ollama') return 'Ollama'
  if (source === 'huggingface_gguf') return 'HF GGUF'
  if (source === 'lmstudio-community') return 'LM Studio'
  return source
}

function sourceColor(source: string): string {
  if (source === 'ollama') return 'badge-primary'
  if (source === 'lmstudio-community') return 'badge-accent'
  return 'badge-secondary'
}

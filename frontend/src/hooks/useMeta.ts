import { useEffect, useState } from 'react'
import { listHarnesses, listUseCases } from '../lib/api'
import type { Harness, UseCase } from '../types'

export function useMeta() {
  const [useCases, setUseCases] = useState<UseCase[]>([])
  const [harnesses, setHarnesses] = useState<Harness[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([listUseCases(), listHarnesses()])
      .then(([uc, hs]) => { setUseCases(uc); setHarnesses(hs) })
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load options'))
      .finally(() => setLoading(false))
  }, [])

  return { useCases, harnesses, loading, error }
}

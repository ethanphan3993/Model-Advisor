import { useState, useCallback } from 'react'
import type { ScanResponse } from '../types'
import { getScan } from '../lib/api'

export function useScan() {
  const [data, setData] = useState<ScanResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const scan = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const result = await getScan()
      setData(result)
      return result
    } catch (e) {
      const msg = e instanceof Error ? e.message : 'Scan failed'
      setError(msg)
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return { data, loading, error, scan, setData }
}

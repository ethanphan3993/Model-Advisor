import { useState, useCallback } from 'react'
import { recommend } from '../lib/api'
import type { Recommendation, HardwareSnapshot, RecommendRequest } from '../types'

export function useRecommend() {
  const [recommendations, setRecommendations] = useState<Recommendation[]>([])
  const [hardware, setHardware] = useState<HardwareSnapshot | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const fetch = useCallback(async (req: RecommendRequest) => {
    setLoading(true)
    setError(null)
    try {
      const result = await recommend(req)
      setRecommendations(result.recommendations)
      setHardware(result.hardware_snapshot)
      return result
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to fetch recommendations')
      return null
    } finally {
      setLoading(false)
    }
  }, [])

  return { recommendations, hardware, loading, error, fetch }
}

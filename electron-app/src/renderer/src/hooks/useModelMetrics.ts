import { useQuery } from '@tanstack/react-query'

export interface FeatureItem {
  feature: string
  importance: number
}

export interface FoldItem {
  fold: number
  accuracy: number
  auc: number
  logLoss: number
}

// /api/model/metrics — written by 03_train_model.py and 04_evaluate_model.py.
// Fields are null until those scripts have run.
export interface ModelMetricsData {
  accuracy: number | null
  baselineAccuracy: number | null
  aucRoc: number | null
  logLoss: number | null
  brierScore: number | null
  featureCount: number | null
  dataPoints: number | null
  nTrials: number | null
  dateFrom: string | null
  dateTo: string | null
  trainedAt: string | null
  evaluatedAt: string | null
  foldMetrics: FoldItem[]
  featureImportance: FeatureItem[]
}

export const API_BASE = 'http://127.0.0.1:8765'

export function useModelMetrics(apiBase: string = API_BASE) {
  return useQuery<ModelMetricsData>({
    queryKey: ['modelMetrics', apiBase],
    queryFn: async () => {
      const res = await fetch(`${apiBase}/api/model/metrics`)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    retry: 1
  })
}

export function formatDateTr(iso: string | null | undefined): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleDateString('tr-TR', {
    day: 'numeric',
    month: 'short',
    year: 'numeric'
  })
}

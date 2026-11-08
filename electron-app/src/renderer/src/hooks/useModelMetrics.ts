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
  headline: 'holdout' | 'oof' | null // which evaluation the headline numbers come from
  holdoutSeason: string | null // "2025-26"
  nHoldoutGames: number | null
  ece: number | null // calibration error on the holdout season
  oofAccuracy: number | null
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
  winSource?: 'classifier' | 'margin'
  lines?: Partial<Record<'margin' | 'total', LineModelMetrics>>
}

// Fark / toplam modelinin holdout özeti (04 → metrics_margin/total.json)
export interface LineModelMetrics {
  mae: number
  baselineMae: number | null
  baseline: string | null
  lowSignal: boolean
  rmse: number
  bias: number
  within10: number
  ladderEce: number | null
  ladderByOffset: { offset: number; meanPred: number; fracOver: number }[]
  distribution: string | null
  bands: { side: 'home' | 'away'; lo: number; hi: number | null; obs: number; pred: number }[]
  nHoldoutGames: number | null
  evaluatedAt: string | null
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

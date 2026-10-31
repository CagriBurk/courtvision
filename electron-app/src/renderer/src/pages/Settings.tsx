import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { RefreshCw, AlertCircle, CheckCircle, Bell } from 'lucide-react'
import { API_BASE, formatDateTr, useModelMetrics } from '../hooks/useModelMetrics'

interface Health {
  status: string
  model_loaded: boolean
  feature_count: number
  live_date: string | null
  live_generated_at: string | null
  updating: boolean
  scheduler_minutes: number
}

function formatDateTimeTr(iso: string | null | undefined): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString('tr-TR', {
    day: 'numeric',
    month: 'short',
    hour: '2-digit',
    minute: '2-digit'
  })
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="text-slate-300 font-semibold text-sm mb-3 uppercase tracking-wide">{title}</h2>
      <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 p-5 space-y-4">
        {children}
      </div>
    </section>
  )
}

function InfoGrid({ rows }: { rows: [string, React.ReactNode][] }) {
  return (
    <div className="grid grid-cols-2 gap-y-2 text-sm">
      {rows.map(([k, v]) => (
        <div key={k} className="flex gap-2">
          <span className="text-slate-500 w-40 shrink-0">{k}</span>
          <span className="text-slate-300">{v}</span>
        </div>
      ))}
    </div>
  )
}

export default function Settings() {
  const { data: health, isError, refetch, isFetching } = useQuery<Health>({
    queryKey: ['health'],
    queryFn: async () => {
      const res = await fetch(`${API_BASE}/api/health`, { signal: AbortSignal.timeout(3000) })
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return res.json()
    },
    retry: 0
  })
  const { data: metrics } = useModelMetrics()

  const [notifPermission, setNotifPermission] = useState(Notification.permission)
  const requestNotif = async () => setNotifPermission(await Notification.requestPermission())

  return (
    <div className="h-full overflow-y-auto">
      <div className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-white font-bold text-xl">Ayarlar</h1>
        <p className="text-slate-500 text-sm mt-0.5">Sunucu, veri ve model durumu</p>
      </div>

      <div className="p-6 space-y-6 max-w-2xl">
        <Section title="Sunucu">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-300 text-sm font-medium">FastAPI</p>
              <p className="text-slate-500 text-xs mt-0.5 font-mono">{API_BASE}</p>
            </div>
            <button
              onClick={() => refetch()}
              className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm px-3 py-2 rounded-lg transition-colors"
            >
              <RefreshCw size={13} className={isFetching ? 'animate-spin' : ''} />
              Test Et
            </button>
          </div>
          {health && (
            <div className="flex items-center gap-1.5 text-emerald-400 text-xs">
              <CheckCircle size={12} /> Bağlantı başarılı
              {!health.model_loaded && (
                <span className="text-amber-400 ml-2">· Model yüklenmedi (03_train_model.py)</span>
              )}
            </div>
          )}
          {isError && (
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-red-400 text-xs">
                <AlertCircle size={12} /> Bağlantı kurulamadı — FastAPI çalışıyor mu?
              </div>
              <code className="block bg-slate-900/50 rounded-lg p-3 text-emerald-400 text-xs font-mono">
                python -m uvicorn src.api_server:app --host 127.0.0.1 --port 8765
              </code>
            </div>
          )}
        </Section>

        <Section title="Veri Güncelleme">
          <InfoGrid
            rows={[
              ['Gösterilen maç günü', formatDateTr(health?.live_date)],
              ['Son güncelleme', formatDateTimeTr(health?.live_generated_at)],
              ['Durum', health?.updating ? 'Güncelleniyor...' : 'Hazır']
            ]}
          />
          <p className="text-slate-500 text-xs leading-relaxed">
            Uygulama açıkken sunucu her {health?.scheduler_minutes ?? 15} dakikada bir yeni bir NBA
            günü başlayıp başlamadığını kontrol eder ve tahminleri kendisi yeniler. Gün içinde elle
            yenilemek için Dashboard&apos;daki &quot;Veriyi Güncelle&quot; butonunu kullanın. Maç
            günü ABD Doğu saatine göre belirlenir.
          </p>
        </Section>

        <Section title="Bildirimler">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-slate-300 text-sm font-medium">Maç öncesi bildirim</p>
              <p className="text-slate-500 text-xs mt-0.5">
                Takip ettiğiniz maçlar için başlamadan 30 dakika önce
              </p>
            </div>
            {notifPermission === 'granted' ? (
              <span className="flex items-center gap-1.5 text-emerald-400 text-xs">
                <CheckCircle size={12} /> İzin verildi
              </span>
            ) : notifPermission === 'denied' ? (
              <span className="text-red-400 text-xs">Engellendi (Windows ayarlarından açın)</span>
            ) : (
              <button
                onClick={requestNotif}
                className="flex items-center gap-2 bg-blue-600 hover:bg-blue-500 text-white text-sm px-3 py-2 rounded-lg transition-colors"
              >
                <Bell size={13} /> İzin Ver
              </button>
            )}
          </div>
        </Section>

        <Section title="Hakkında">
          <InfoGrid
            rows={[
              ['Versiyon', 'v1.0.0'],
              ['Model', 'XGBoost + Optuna'],
              ['Feature sayısı', metrics?.featureCount ?? '—'],
              ['Veri kaynağı', 'nba_api (stats.nba.com)'],
              [
                'Eğitim verisi',
                metrics?.dataPoints
                  ? `${metrics.dataPoints.toLocaleString('tr-TR')} maç (${metrics.dateFrom?.slice(0, 4)}–${metrics.dateTo?.slice(0, 4)})`
                  : '—'
              ],
              ['Son eğitim', formatDateTr(metrics?.trainedAt)],
              ['OOF doğruluk', metrics?.accuracy != null ? `%${metrics.accuracy}` : '—']
            ]}
          />
        </Section>
      </div>
    </div>
  )
}

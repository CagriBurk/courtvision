import { useState } from 'react'
import { Save, RefreshCw, AlertCircle, CheckCircle } from 'lucide-react'

export default function Settings() {
  const [apiUrl, setApiUrl] = useState('http://127.0.0.1:8765')
  const [autoUpdate, setAutoUpdate] = useState(true)
  const [updateTime, setUpdateTime] = useState('10:00')
  const [seasonFilter, setSeasonFilter] = useState('Regular Season')
  const [saved, setSaved] = useState(false)
  const [apiStatus, setApiStatus] = useState<'idle' | 'ok' | 'error'>('idle')

  const handleSave = () => {
    setSaved(true)
    setTimeout(() => setSaved(false), 2000)
  }

  const testConnection = async () => {
    setApiStatus('idle')
    try {
      const res = await fetch(`${apiUrl}/api/health`, { signal: AbortSignal.timeout(3000) })
      setApiStatus(res.ok ? 'ok' : 'error')
    } catch {
      setApiStatus('error')
    }
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="border-b border-slate-800 px-6 py-4">
        <h1 className="text-white font-bold text-xl">Ayarlar</h1>
        <p className="text-slate-500 text-sm mt-0.5">API bağlantısı ve güncelleme tercihleri</p>
      </div>

      <div className="p-6 space-y-6 max-w-2xl">
        {/* API Settings */}
        <section>
          <h2 className="text-slate-300 font-semibold text-sm mb-3 uppercase tracking-wide">
            FastAPI Bağlantısı
          </h2>
          <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 p-5 space-y-4">
            <div>
              <label className="text-slate-400 text-xs font-medium mb-1.5 block">
                API Endpoint URL
              </label>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={apiUrl}
                  onChange={(e) => setApiUrl(e.target.value)}
                  className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500 transition-colors"
                />
                <button
                  onClick={testConnection}
                  className="flex items-center gap-2 bg-slate-700 hover:bg-slate-600 text-slate-300 text-sm px-3 py-2 rounded-lg transition-colors whitespace-nowrap"
                >
                  <RefreshCw size={13} />
                  Test Et
                </button>
              </div>
              {apiStatus === 'ok' && (
                <div className="flex items-center gap-1.5 mt-1.5 text-emerald-400 text-xs">
                  <CheckCircle size={12} /> Bağlantı başarılı
                </div>
              )}
              {apiStatus === 'error' && (
                <div className="flex items-center gap-1.5 mt-1.5 text-red-400 text-xs">
                  <AlertCircle size={12} /> Bağlantı kurulamadı — FastAPI çalışıyor mu?
                </div>
              )}
            </div>

            <div className="bg-slate-900/50 rounded-lg p-3 text-xs text-slate-500">
              <p className="font-medium text-slate-400 mb-1">Sunucuyu Başlatmak İçin:</p>
              <code className="text-emerald-400 font-mono">
                uvicorn src.api_server:app --host 127.0.0.1 --port 8765
              </code>
            </div>
          </div>
        </section>

        {/* Auto-update Settings */}
        <section>
          <h2 className="text-slate-300 font-semibold text-sm mb-3 uppercase tracking-wide">
            Otomatik Güncelleme
          </h2>
          <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 p-5 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-slate-300 text-sm font-medium">Günlük Otomatik Güncelleme</p>
                <p className="text-slate-500 text-xs mt-0.5">
                  nba_api'den veri çekip tahminleri yeniler
                </p>
              </div>
              <button
                onClick={() => setAutoUpdate(!autoUpdate)}
                className={`relative w-11 h-6 rounded-full transition-colors ${
                  autoUpdate ? 'bg-blue-600' : 'bg-slate-700'
                }`}
              >
                <div
                  className={`absolute top-1 w-4 h-4 bg-white rounded-full transition-transform ${
                    autoUpdate ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>

            {autoUpdate && (
              <div>
                <label className="text-slate-400 text-xs font-medium mb-1.5 block">
                  Güncelleme Saati (TR)
                </label>
                <input
                  type="time"
                  value={updateTime}
                  onChange={(e) => setUpdateTime(e.target.value)}
                  className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500 transition-colors"
                />
                <p className="text-slate-600 text-xs mt-1">
                  Tavsiye: 10:00–11:00 TR (injury report açıklanmış olur)
                </p>
              </div>
            )}
          </div>
        </section>

        {/* Data Settings */}
        <section>
          <h2 className="text-slate-300 font-semibold text-sm mb-3 uppercase tracking-wide">
            Veri Ayarları
          </h2>
          <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 p-5 space-y-4">
            <div>
              <label className="text-slate-400 text-xs font-medium mb-1.5 block">
                Sezon Tipi
              </label>
              <select
                value={seasonFilter}
                onChange={(e) => setSeasonFilter(e.target.value)}
                className="bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-white text-sm focus:outline-none focus:border-blue-500 transition-colors w-full"
              >
                <option>Regular Season</option>
                <option>Regular Season + Playoffs</option>
              </select>
              <p className="text-slate-600 text-xs mt-1">
                Playoffs farklı dinamikler içerdiğinden eğitimden hariç tutulması tavsiye edilir
              </p>
            </div>
          </div>
        </section>

        {/* About */}
        <section>
          <h2 className="text-slate-300 font-semibold text-sm mb-3 uppercase tracking-wide">
            Hakkında
          </h2>
          <div className="bg-slate-800/40 rounded-xl border border-slate-700/40 p-5">
            <div className="grid grid-cols-2 gap-y-2 text-sm">
              {[
                ['Versiyon', 'v1.0.0'],
                ['Model', 'XGBoost + Optuna'],
                ['Feature Sayısı', '236'],
                ['Veri Kaynağı', 'Kaggle + nba_api'],
                ['Eğitim Verisi', '26,332 maç (2000–2026)'],
                ['Son Model', '22 Şub 2026'],
              ].map(([k, v]) => (
                <div key={k} className="flex gap-2">
                  <span className="text-slate-500 w-36">{k}</span>
                  <span className="text-slate-300">{v}</span>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* Save button */}
        <button
          onClick={handleSave}
          className={`flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold transition-all ${
            saved
              ? 'bg-emerald-600 text-white'
              : 'bg-blue-600 hover:bg-blue-500 text-white'
          }`}
        >
          {saved ? <CheckCircle size={15} /> : <Save size={15} />}
          {saved ? 'Kaydedildi!' : 'Kaydet'}
        </button>
      </div>
    </div>
  )
}

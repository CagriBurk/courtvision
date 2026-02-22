import { useState } from 'react'
import { LayoutDashboard, History, BarChart2, Settings } from 'lucide-react'
import { NavPage } from '../types'
import { NBA_LOGO } from '../data/mockData'

interface SidebarProps {
  activePage: NavPage
  onNavigate: (page: NavPage) => void
}

const navItems = [
  { page: 'dashboard' as NavPage, label: 'Dashboard', icon: LayoutDashboard },
  { page: 'history' as NavPage, label: 'Geçmiş', icon: History },
  { page: 'metrics' as NavPage, label: 'Model Metrikleri', icon: BarChart2 },
  { page: 'settings' as NavPage, label: 'Ayarlar', icon: Settings },
]

export default function Sidebar({ activePage, onNavigate }: SidebarProps) {
  const [logoError, setLogoError] = useState(false)

  return (
    <aside className="w-52 bg-slate-900 border-r border-slate-800 flex flex-col shrink-0">
      {/* Logo */}
      <div className="p-4 border-b border-slate-800">
        <div className="flex items-center gap-2.5">
          {!logoError ? (
            <img
              src={NBA_LOGO}
              alt="NBA"
              className="w-9 h-9 object-contain"
              onError={() => setLogoError(true)}
            />
          ) : (
            <div className="w-9 h-9 bg-blue-600 rounded-lg flex items-center justify-center font-bold text-white text-xs">
              NBA
            </div>
          )}
          <div>
            <p className="text-white text-sm font-bold leading-none">CourtVision</p>
            <p className="text-slate-500 text-xs mt-0.5">AI Tahmin Sistemi</p>
          </div>
        </div>
      </div>

      {/* Nav items */}
      <nav className="flex-1 p-3 space-y-1">
        {navItems.map(({ page, label, icon: Icon }) => (
          <button
            key={page}
            onClick={() => onNavigate(page)}
            className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors text-left ${
              activePage === page
                ? 'bg-blue-600 text-white'
                : 'text-slate-400 hover:bg-slate-800 hover:text-white'
            }`}
          >
            <Icon size={16} />
            {label}
          </button>
        ))}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-slate-800">
        <p className="text-slate-600 text-xs">v1.0.0</p>
        <p className="text-slate-700 text-xs">236 feature</p>
      </div>
    </aside>
  )
}

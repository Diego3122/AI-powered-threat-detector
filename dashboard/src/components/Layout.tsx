import { useMemo, useState } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import {
  Activity,
  BrainCircuit,
  LayoutDashboard,
  Menu,
  ScrollText,
  ShieldAlert,
  X,
} from 'lucide-react'

import { useAuth } from '@/auth/AuthProvider'

export default function Layout() {
  const { user, logout } = useAuth()
  const [navOpen, setNavOpen] = useState(true)
  const location = useLocation()

  const navItems = [
    { path: '/', label: 'Dashboard', icon: LayoutDashboard },
    { path: '/incidents', label: 'Threats', icon: ShieldAlert },
    { path: '/models', label: 'Models', icon: BrainCircuit },
    { path: '/logs', label: 'Logs', icon: ScrollText },
  ]

  const isActive = (path: string) => location.pathname === path
  const activePage = useMemo(
    () => navItems.find((item) => isActive(item.path)) ?? navItems[0],
    [location.pathname]
  )
  const contentOffset = navOpen ? 'lg:ml-72' : 'lg:ml-20'

  return (
    <div className="min-h-screen bg-sentinel-bg text-sentinel-text">
      <div className="pointer-events-none fixed inset-0 opacity-40">
        <div className="sentinel-subtle-grid absolute inset-0" />
      </div>

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex flex-col border-r border-sentinel-border/40 bg-sentinel-low/95 shadow-panel backdrop-blur-md transition-all duration-300 ${
          navOpen ? 'w-72' : 'w-20'
        }`}
      >
        <div className="flex h-20 items-center justify-between border-b border-sentinel-border/30 px-4">
          <div className="flex items-center gap-3 overflow-hidden">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-sentinel-accentStrong/15 text-sentinel-accent shadow-glow">
              <ShieldAlert size={20} strokeWidth={1.8} />
            </div>
          </div>
          <button
            onClick={() => setNavOpen(!navOpen)}
            className="rounded-md border border-sentinel-border/40 bg-sentinel-panel/60 p-2 text-sentinel-muted hover:border-sentinel-accent/40 hover:text-sentinel-accent"
          >
            {navOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>

        <div className="flex-1 px-3 py-6">
          <div className={`mb-4 ${navOpen ? 'px-3' : 'px-0 text-center'}`}>
            <p className="font-mono text-[10px] uppercase tracking-[0.24em] text-neutral-subtext">
              {navOpen ? 'Navigation' : 'Nav'}
            </p>
          </div>

          <div className="space-y-2">
            {navItems.map((item) => (
              <Link
                key={item.path}
                to={item.path}
                className={`flex items-center gap-3 rounded-xl px-4 py-3 text-sm transition-all ${
                  isActive(item.path)
                    ? 'border border-sentinel-accent/25 bg-sentinel-panel text-sentinel-accent shadow-glow'
                    : 'border border-transparent text-neutral-subtext hover:border-sentinel-border/30 hover:bg-sentinel-panel/70 hover:text-neutral-white'
                }`}
              >
                <item.icon size={18} strokeWidth={1.8} />
                {navOpen && <span>{item.label}</span>}
              </Link>
            ))}
          </div>
        </div>

        <div className="border-t border-sentinel-border/30 p-4">
          <div className={`rounded-xl border border-sentinel-border/30 bg-sentinel-panel/80 p-3 ${navOpen ? '' : 'text-center'}`}>
            <div className={`flex items-center gap-3 ${navOpen ? '' : 'justify-center'}`}>
              <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-sentinel-accentStrong/10 text-sentinel-accent">
                <Activity size={16} />
              </div>
              {navOpen && (
                <div>
                  <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-sentinel-accent">
                    Live Engine Active
                  </p>
                  <p className="text-xs text-neutral-subtext">CPU runtime / telemetry online</p>
                </div>
              )}
            </div>
          </div>
        </div>
      </aside>

      <main className={`relative min-h-screen transition-all duration-300 ${contentOffset}`}>
        <header className="sticky top-0 z-30 border-b border-sentinel-border/20 bg-sentinel-bg/80 backdrop-blur-xl">
          <div className="mx-auto flex max-w-[1600px] items-center justify-between gap-6 px-6 py-5 md:px-8">
            <div className="space-y-1">
              <p className="font-mono text-[10px] uppercase tracking-[0.28em] text-neutral-subtext">
                Analysis / {activePage.label}
              </p>
              <div className="flex items-center gap-3">
                <h2 className="font-editorial text-2xl font-bold uppercase tracking-tight text-neutral-white">
                  {activePage.label}
                </h2>
                <span className="hidden rounded-full border border-sentinel-accent/20 bg-sentinel-accentStrong/10 px-3 py-1 font-mono text-[10px] uppercase tracking-[0.2em] text-sentinel-accent md:inline-flex">
                  Live
                </span>
              </div>
            </div>

            <div className="flex items-center gap-3">
              <div className="hidden rounded-xl border border-sentinel-border/30 bg-sentinel-panel/70 px-4 py-3 lg:block">
                <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-neutral-subtext">Authenticated User</p>
                <p className="mt-1 text-sm text-neutral-white">{user?.username ?? 'unknown'}</p>
              </div>
              <div className="hidden rounded-xl border border-sentinel-border/30 bg-sentinel-panel/70 px-4 py-3 md:block">
                <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-neutral-subtext">Engine Status</p>
                <p className="mt-1 text-sm text-neutral-white">Streaming detections online</p>
              </div>
              <div className="rounded-xl border border-sentinel-border/30 bg-sentinel-panel/70 px-4 py-3">
                <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-neutral-subtext">Current Session</p>
                <p className="mt-1 text-sm text-neutral-white">
                  {new Date().toLocaleDateString('en-US', {
                    weekday: 'short',
                    month: 'short',
                    day: 'numeric',
                  })}
                </p>
              </div>
              <button
                onClick={logout}
                className="rounded-xl border border-sentinel-border/30 bg-sentinel-panel/70 px-4 py-3 font-mono text-[10px] uppercase tracking-[0.2em] text-neutral-subtext transition hover:border-sentinel-accent/40 hover:text-sentinel-accent"
              >
                Sign Out
              </button>
            </div>
          </div>
        </header>

        <div className="mx-auto max-w-[1600px] px-6 py-8 md:px-8">
          <Outlet />
        </div>

        <footer className="border-t border-sentinel-border/20 px-6 py-4 md:px-8">
          <div className="mx-auto flex max-w-[1600px] flex-col gap-3 font-mono text-[10px] uppercase tracking-[0.18em] text-neutral-subtext md:flex-row md:items-center md:justify-between">
            <div className="flex flex-wrap items-center gap-4">
              <span className="inline-flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-sentinel-accent" />
                Core DB Synced
              </span>
              <span className="inline-flex items-center gap-2">
                <span className="h-2 w-2 rounded-full bg-sentinel-success" />
                API Mesh Online
              </span>
            </div>
            <span className="text-sentinel-accent">Threat Detector</span>
          </div>
        </footer>
      </main>
    </div>
  )
}

import { FormEvent, useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import axios from 'axios'

import { useAuth } from '@/auth/AuthProvider'

export default function Login() {
  const { user, login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const [username, setUsername] = useState('analyst')
  const [password, setPassword] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (user) {
      const destination = (location.state as { from?: { pathname?: string } } | null)?.from?.pathname || '/'
      navigate(destination, { replace: true })
    }
  }, [user, location.state, navigate])

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    setSubmitting(true)
    setError(null)

    try {
      await login(username, password)
      const destination = (location.state as { from?: { pathname?: string } } | null)?.from?.pathname || '/'
      navigate(destination, { replace: true })
    } catch (submitError) {
      if (axios.isAxiosError(submitError)) {
        setError(submitError.response?.data?.detail || 'Sign in failed')
      } else {
        setError('Sign in failed')
      }
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-sentinel-bg px-6 py-12 text-sentinel-text">
      <div className="pointer-events-none fixed inset-0 opacity-40">
        <div className="sentinel-subtle-grid absolute inset-0" />
      </div>

      <div className="sentinel-card relative w-full max-w-md p-8 shadow-glow">
        <div className="space-y-3">
          <p className="sentinel-kicker">Secure access</p>
          <h1 className="font-editorial text-4xl font-bold uppercase tracking-tight text-neutral-white">
            Threat Detector
          </h1>
          <p className="text-sm text-neutral-subtext">
            Sign in to view alerts, models, investigations, and audit activity.
          </p>
        </div>

        <form className="mt-8 space-y-5" onSubmit={handleSubmit}>
          <label className="block space-y-2">
            <span className="text-quiet">Username</span>
            <input
              className="w-full rounded-xl border border-sentinel-border/40 bg-sentinel-panel/80 px-4 py-3 text-sm text-neutral-white outline-none transition focus:border-sentinel-accent/50"
              value={username}
              onChange={(event) => setUsername(event.target.value)}
              autoComplete="username"
              required
            />
          </label>

          <label className="block space-y-2">
            <span className="text-quiet">Password</span>
            <input
              className="w-full rounded-xl border border-sentinel-border/40 bg-sentinel-panel/80 px-4 py-3 text-sm text-neutral-white outline-none transition focus:border-sentinel-accent/50"
              type="password"
              value={password}
              onChange={(event) => setPassword(event.target.value)}
              autoComplete="current-password"
              required
            />
          </label>

          {error && (
            <div className="rounded-xl border border-threat-deep/30 bg-threat-bg/70 px-4 py-3 text-sm text-threat-muted">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={submitting}
            className="inline-flex w-full items-center justify-center rounded-xl bg-sentinel-accentStrong px-4 py-3 font-mono text-[11px] uppercase tracking-[0.18em] text-sentinel-bg transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {submitting ? 'Signing In...' : 'Sign In'}
          </button>
        </form>
      </div>
    </div>
  )
}

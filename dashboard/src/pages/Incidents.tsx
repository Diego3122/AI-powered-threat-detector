import { useCallback, useEffect, useRef, useState } from 'react'
import { getStoredToken, threatAPI } from '@/api/client'
import AlertDetail from '@/components/AlertDetail'

const API_BASE = (() => {
  const configured = import.meta.env.VITE_API_URL?.trim()
  if (configured) return configured.replace(/\/+$/, '')
  if (typeof window !== 'undefined') return window.location.origin
  return 'http://localhost:8000'
})()

interface Alert {
  id: number
  timestamp: number
  window_id: string
  model_type: string
  feature_schema?: string
  model_score: number
  threshold: number
  triggered: boolean
  explanation_summary?: string
  created_at: string
}

export default function Incidents() {
  const [incidents, setIncidents] = useState<Alert[]>([])
  const [loading, setLoading] = useState(true)
  const [live, setLive] = useState(false)
  const [selectedAlert, setSelectedAlert] = useState<Alert | null>(null)
  const maxIdRef = useRef(0)
  const abortRef = useRef<AbortController | null>(null)

  // Merge new alerts into state without duplicates, newest first
  const mergeAlerts = useCallback((incoming: Alert[]) => {
    setIncidents((prev) => {
      const byId = new Map(prev.map((a) => [a.id, a]))
      for (const a of incoming) byId.set(a.id, a)
      return [...byId.values()].sort((a, b) => b.id - a.id)
    })
    for (const a of incoming) {
      if (a.id > maxIdRef.current) maxIdRef.current = a.id
    }
  }, [])

  // Initial load via REST
  useEffect(() => {
    let active = true
    threatAPI
      .getIncidents()
      .then((data) => {
        if (active && Array.isArray(data)) {
          mergeAlerts(data)
        }
      })
      .catch((err) => console.error('Failed to load incidents:', err))
      .finally(() => {
        if (active) setLoading(false)
      })
    return () => {
      active = false
    }
  }, [mergeAlerts])

  // SSE connection with polling fallback
  useEffect(() => {
    let fallbackInterval: number | null = null

    const connectSSE = () => {
      const token = getStoredToken()
      const controller = new AbortController()
      abortRef.current = controller

      fetch(`${API_BASE}/api/alerts/stream?since_id=${maxIdRef.current}`, {
        headers: token ? { Authorization: `Bearer ${token}` } : {},
        signal: controller.signal,
      })
        .then(async (response) => {
          if (!response.ok || !response.body) throw new Error(`SSE ${response.status}`)
          setLive(true)

          const reader = response.body.getReader()
          const decoder = new TextDecoder()
          let buffer = ''

          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            buffer += decoder.decode(value, { stream: true })

            const lines = buffer.split('\n')
            buffer = lines.pop() ?? ''

            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const alert: Alert = JSON.parse(line.slice(6))
                  mergeAlerts([alert])
                } catch {
                  // ignore malformed event
                }
              }
            }
          }

          // Stream ended cleanly — reconnect after a short delay
          setLive(false)
          if (!controller.signal.aborted) {
            setTimeout(connectSSE, 5000)
          }
        })
        .catch((err) => {
          if (err.name === 'AbortError') return
          console.warn('SSE unavailable, falling back to polling:', err)
          setLive(false)
          // Fallback: poll every 10 s
          fallbackInterval = window.setInterval(async () => {
            try {
              const data = await threatAPI.getIncidents()
              if (Array.isArray(data)) mergeAlerts(data)
            } catch {
              // ignore
            }
          }, 10000)
        })
    }

    connectSSE()

    return () => {
      abortRef.current?.abort()
      abortRef.current = null
      if (fallbackInterval !== null) window.clearInterval(fallbackInterval)
      setLive(false)
    }
  }, [mergeAlerts])

  const getSeverityColor = (score: number) => {
    if (score > 0.8) return 'threat'
    if (score > 0.6) return 'anomaly'
    return 'insight'
  }

  const getTimeAgo = (timestamp: number) => {
    const now = Date.now()
    const diff = now - timestamp
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)
    const days = Math.floor(diff / 86400000)

    if (minutes < 60) return `${minutes}m ago`
    if (hours < 24) return `${hours}h ago`
    return `${days}d ago`
  }

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <p className="sentinel-kicker">Threat intelligence / queue</p>
        <div className="flex items-center gap-3">
          <h1 className="heading-editorial">Incidents</h1>
          <span
            className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-[10px] uppercase tracking-[0.18em] ${
              live
                ? 'border-insight-blue/30 bg-insight-bg text-insight-light'
                : 'border-sentinel-border/30 bg-sentinel-elevated/40 text-neutral-subtext'
            }`}
          >
            <span
              className={`h-1.5 w-1.5 rounded-full ${live ? 'bg-insight-light animate-pulse' : 'bg-neutral-subtext'}`}
            />
            {live ? 'Live' : 'Polling'}
          </span>
        </div>
        <p className="max-w-2xl text-sm text-neutral-subtext">
          Detailed incident feed rendered as a high-contrast operator board. Severity, confidence, and context remain unchanged.
        </p>
      </div>

      {loading ? (
        <div className="sentinel-card p-10 text-center text-neutral-subtext">
          <p>Loading incidents...</p>
        </div>
      ) : incidents.length === 0 ? (
        <div className="sentinel-card p-10 text-center text-neutral-subtext">
          <p>No incidents detected</p>
          <p className="mt-2 text-sm">All systems operating normally</p>
        </div>
      ) : (
        <div className="space-y-4">
          {incidents.map((incident) => (
            <button
              key={incident.id}
              className="w-full text-left sentinel-card overflow-hidden hover:border-sentinel-accent/40 cursor-pointer"
              onClick={() => setSelectedAlert(incident)}
            >
              <div className="h-1 w-full bg-sentinel-border/20">
                <div
                  className={`h-full ${
                    getSeverityColor(incident.model_score) === 'threat'
                      ? 'bg-threat-deep'
                      : getSeverityColor(incident.model_score) === 'anomaly'
                        ? 'bg-anomaly-accent'
                        : 'bg-sentinel-accent'
                  }`}
                  style={{ width: `${incident.model_score * 100}%` }}
                />
              </div>
              <div className="p-6">
                <div className="mb-4 flex items-start justify-between">
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-3">
                      <span
                        className={`${
                          getSeverityColor(incident.model_score) === 'threat'
                            ? 'threat-badge'
                            : getSeverityColor(incident.model_score) === 'anomaly'
                              ? 'anomaly-badge'
                              : 'insight-badge'
                        }`}
                      >
                        {getSeverityColor(incident.model_score) === 'threat'
                          ? 'Threat'
                          : getSeverityColor(incident.model_score) === 'anomaly'
                            ? 'Anomaly'
                            : 'Insight'}
                      </span>
                      <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-neutral-subtext">
                        {getTimeAgo(incident.timestamp)}
                      </span>
                    </div>
                    <h3 className="text-lg font-medium text-neutral-white">
                      {incident.explanation_summary || `Detection by ${incident.model_type}`}
                    </h3>
                  </div>
                  <div className="text-right">
                    <div className="font-editorial text-3xl font-bold text-neutral-white">
                      {Math.round(incident.model_score * 100)}%
                    </div>
                    <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-neutral-subtext">Confidence</div>
                  </div>
                </div>

                <div className="space-y-3">
                  <div className="h-2 w-full overflow-hidden rounded-full bg-sentinel-border/20">
                    <div
                      className={`h-full rounded-full ${
                        getSeverityColor(incident.model_score) === 'threat'
                          ? 'bg-threat-deep'
                          : getSeverityColor(incident.model_score) === 'anomaly'
                            ? 'bg-anomaly-accent'
                            : 'bg-insight-blue'
                      }`}
                      style={{ width: `${incident.model_score * 100}%` }}
                    />
                  </div>
                  <div className="flex flex-wrap justify-between gap-3 font-mono text-[10px] uppercase tracking-[0.18em] text-neutral-subtext">
                    <span>Model: {incident.model_type}</span>
                    <span>Click to investigate →</span>
                    <span>{new Date(incident.created_at).toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </button>
          ))}
        </div>
      )}

      {selectedAlert && (
        <AlertDetail
          alert={selectedAlert}
          onClose={() => setSelectedAlert(null)}
        />
      )}
    </div>
  )
}

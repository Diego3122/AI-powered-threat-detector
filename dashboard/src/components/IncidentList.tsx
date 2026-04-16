import { useEffect, useState } from 'react'
import { threatAPI } from '@/api/client'

interface Incident {
  id: number
  timestamp: number
  model_type: string
  model_score: number
  triggered: boolean
  explanation_summary?: string
  created_at: string
}

export default function IncidentList() {
  const [incidents, setIncidents] = useState<Incident[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true

    const fetchIncidents = async () => {
      try {
        const data = await threatAPI.getIncidents()
        if (active && data) {
          setIncidents(Array.isArray(data) ? data.slice(0, 3) : [])
        }
      } catch (error) {
        console.error('Failed to load incidents:', error)
      } finally {
        if (active) setLoading(false)
      }
    }

    fetchIncidents()
    const intervalId = window.setInterval(fetchIncidents, 10000)

    return () => {
      active = false
      window.clearInterval(intervalId)
    }
  }, [])

  if (loading) {
    return <div className="text-sm text-neutral-subtext">Loading incidents...</div>
  }

  if (incidents.length === 0) {
    return <div className="sentinel-card-soft p-4 text-sm text-neutral-subtext">No active incidents</div>
  }

  const getTimeAgo = (timestamp: number) => {
    const now = Date.now()
    const diff = now - timestamp
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)

    if (minutes < 60) return `${minutes}m ago`
    if (hours < 24) return `${hours}h ago`
    return new Date(timestamp).toLocaleDateString()
  }

  const getSeverity = (score: number) => {
    if (score > 0.8) return 'threat'
    if (score > 0.6) return 'anomaly'
    return 'anomaly'
  }

  return (
    <div className="space-y-3">
      {incidents.map((incident) => (
        <div key={incident.id} className="sentinel-card-soft p-4">
          <div className="flex items-start justify-between gap-4">
            <div className="flex-1 space-y-2">
              <div className="flex items-center gap-2">
                <span className={`${getSeverity(incident.model_score) === 'threat' ? 'threat-badge' : 'anomaly-badge'}`}>
                  {getSeverity(incident.model_score) === 'threat' ? 'Threat' : 'Anomaly'}
                </span>
                <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-neutral-subtext">
                  {getTimeAgo(incident.timestamp)}
                </span>
              </div>
              <p className="text-sm font-medium text-neutral-white">
                {incident.explanation_summary || `Detection by ${incident.model_type} model`}
              </p>
              <div className="flex flex-wrap items-center gap-4">
                <div className="flex items-center gap-2">
                  <div className="h-1.5 w-24 overflow-hidden rounded-full bg-sentinel-border/30">
                    <div
                      className={`h-full rounded-full ${getSeverity(incident.model_score) === 'threat' ? 'bg-threat-deep' : 'bg-anomaly-accent'}`}
                      style={{ width: `${Math.round(incident.model_score * 100)}%` }}
                    />
                  </div>
                  <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-neutral-subtext">
                    {Math.round(incident.model_score * 100)}%
                  </span>
                </div>
                <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-neutral-subtext">
                  {incident.model_type}
                </span>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}

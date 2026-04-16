import { useEffect, useState } from 'react'
import { threatAPI } from '@/api/client'

interface AuditLog {
  id: number
  user_id: string
  action: string
  target: string
  details?: string
  created_at: string
}

export default function Logs() {
  const [logs, setLogs] = useState<AuditLog[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true

    const fetchLogs = async () => {
      try {
        const data = await threatAPI.getAuditLogs(100, 0)
        if (active && data) {
          setLogs(Array.isArray(data) ? data : [])
        }
      } catch (error) {
        console.error('Failed to load logs:', error)
      } finally {
        if (active) setLoading(false)
      }
    }

    fetchLogs()
    const intervalId = window.setInterval(fetchLogs, 10000)

    return () => {
      active = false
      window.clearInterval(intervalId)
    }
  }, [])

  const getTimeAgo = (timestamp: string) => {
    const now = new Date()
    const diff = now.getTime() - new Date(timestamp).getTime()
    const minutes = Math.floor(diff / 60000)
    const hours = Math.floor(diff / 3600000)
    const days = Math.floor(diff / 86400000)

    if (minutes < 60) return `${minutes}m ago`
    if (hours < 24) return `${hours}h ago`
    return `${days}d ago`
  }

  const getActionColor = (action: string) => {
    if (action.includes('triggered') || action.includes('alert')) return 'threat'
    if (action.includes('error') || action.includes('fail')) return 'anomaly'
    return 'insight'
  }

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <p className="sentinel-kicker">Audit trail / system activity</p>
        <h1 className="heading-editorial">Logs</h1>
        <p className="max-w-2xl text-sm text-neutral-subtext">
          Raw system activity and audit events presented with a terminal-inspired visual treatment.
        </p>
      </div>

      {loading ? (
        <div className="sentinel-card p-10 text-center text-neutral-subtext">
          <p>Loading logs...</p>
        </div>
      ) : logs.length === 0 ? (
        <div className="sentinel-card p-10 text-center text-neutral-subtext">
          <p>No logs available</p>
        </div>
      ) : (
        <div className="sentinel-card overflow-hidden">
          {logs.map((log) => (
            <div key={log.id} className="sentinel-table-row px-5 py-4 first:border-t-0">
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-3">
                    <span
                      className={`${
                        getActionColor(log.action) === 'threat'
                          ? 'threat-badge'
                          : getActionColor(log.action) === 'anomaly'
                            ? 'anomaly-badge'
                            : 'insight-badge'
                      }`}
                    >
                      {log.action}
                    </span>
                    <span className="font-mono text-[10px] uppercase tracking-[0.18em] text-neutral-subtext">
                      {getTimeAgo(log.created_at)}
                    </span>
                  </div>
                  <div className="grid gap-3 text-sm md:grid-cols-3">
                    <div>
                      <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-neutral-subtext">User </span>
                      <span className="font-medium text-neutral-white">{log.user_id || 'system'}</span>
                    </div>
                    <div>
                      <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-neutral-subtext">Target </span>
                      <span className="font-medium text-neutral-white">{log.target}</span>
                    </div>
                    <div>
                      <span className="font-mono text-[10px] uppercase tracking-[0.16em] text-neutral-subtext">Action </span>
                      <span className="font-medium text-neutral-white">{log.action}</span>
                    </div>
                  </div>
                  {log.details && <div className="sentinel-terminal mt-2">{log.details}</div>}
                </div>
                <div className="font-mono text-[10px] uppercase tracking-[0.18em] text-neutral-subtext">
                  {new Date(log.created_at).toLocaleTimeString()}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

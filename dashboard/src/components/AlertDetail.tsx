import { useEffect, useRef, useState } from 'react'
import { threatAPI } from '@/api/client'

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

interface Investigation {
  id: number
  alert_id: number
  user_id?: string
  status: string
  notes?: string
  created_at: string
  updated_at: string
}

const STATUS_LABELS: Record<string, string> = {
  open: 'Open',
  investigating: 'Investigating',
  resolved: 'Resolved',
  false_positive: 'False Positive',
}

const STATUS_COLORS: Record<string, string> = {
  open: 'text-anomaly-light border-anomaly-accent/30 bg-anomaly-bg',
  investigating: 'text-sentinel-accent border-sentinel-accent/30 bg-sentinel-panel',
  resolved: 'text-insight-light border-insight-blue/30 bg-insight-bg',
  false_positive: 'text-neutral-subtext border-sentinel-border/30 bg-sentinel-elevated/40',
}

interface Props {
  alert: Alert
  onClose: () => void
}

export default function AlertDetail({ alert, onClose }: Props) {
  const [investigations, setInvestigations] = useState<Investigation[]>([])
  const [loadingInv, setLoadingInv] = useState(true)
  const [notes, setNotes] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const panelRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    let active = true
    setLoadingInv(true)
    threatAPI
      .getAlertInvestigations(alert.id)
      .then((data) => {
        if (active) setInvestigations(Array.isArray(data) ? data : [])
      })
      .catch(() => {
        if (active) setInvestigations([])
      })
      .finally(() => {
        if (active) setLoadingInv(false)
      })
    return () => {
      active = false
    }
  }, [alert.id])

  // Close on Escape
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', handler)
    return () => document.removeEventListener('keydown', handler)
  }, [onClose])

  const getSeverityColor = (score: number) => {
    if (score > 0.8) return 'threat'
    if (score > 0.6) return 'anomaly'
    return 'insight'
  }

  const severity = getSeverityColor(alert.model_score)

  const scoreBarColor =
    severity === 'threat'
      ? 'bg-threat-deep'
      : severity === 'anomaly'
        ? 'bg-anomaly-accent'
        : 'bg-insight-blue'

  const currentStatus =
    investigations.length > 0 ? investigations[investigations.length - 1].status : null

  const handleAction = async (status: string) => {
    setSubmitting(true)
    setError(null)
    try {
      const existing = investigations.find((i) => i.status === status)
      if (existing) {
        const updated = await threatAPI.updateInvestigation(existing.id, {
          status,
          notes: notes || undefined,
        })
        setInvestigations((prev) => prev.map((i) => (i.id === updated.id ? updated : i)))
      } else {
        const created = await threatAPI.createInvestigation(alert.id, {
          status,
          notes: notes || undefined,
        })
        setInvestigations((prev) => [...prev, created])
      }
      setNotes('')
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Action failed')
    } finally {
      setSubmitting(false)
    }
  }

  // Parse explanation_summary into key-value pairs for display
  const explainParts = alert.explanation_summary
    ? alert.explanation_summary.split(' | ').map((part) => {
        const eqIdx = part.indexOf('=')
        if (eqIdx === -1) return { key: part, value: '' }
        return { key: part.slice(0, eqIdx), value: part.slice(eqIdx + 1) }
      })
    : []

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Slide-over panel */}
      <div
        ref={panelRef}
        className="fixed inset-y-0 right-0 z-50 flex w-full max-w-xl flex-col overflow-hidden border-l border-sentinel-border/60 bg-sentinel-panel shadow-2xl"
        role="dialog"
        aria-modal="true"
        aria-label="Alert detail"
      >
        {/* Severity stripe */}
        <div
          className={`h-1 w-full ${scoreBarColor}`}
        />

        {/* Header */}
        <div className="flex items-start justify-between border-b border-sentinel-border/30 p-6">
          <div className="space-y-1">
            <p className="sentinel-kicker">Alert #{alert.id}</p>
            <h2 className="text-lg font-semibold text-neutral-white">
              {alert.explanation_summary
                ? alert.explanation_summary.split(' | ')[0]
                : `Detection by ${alert.model_type}`}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="ml-4 rounded-md p-1 text-neutral-subtext hover:text-neutral-white"
            aria-label="Close"
          >
            <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Scrollable body */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">

          {/* Score gauge */}
          <div className="sentinel-card-soft p-4 space-y-2">
            <div className="flex justify-between items-center">
              <span className="text-quiet">Confidence Score</span>
              <span className="font-editorial text-2xl font-bold text-neutral-white">
                {Math.round(alert.model_score * 100)}%
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-sentinel-border/20">
              <div
                className={`h-full rounded-full ${scoreBarColor}`}
                style={{ width: `${alert.model_score * 100}%` }}
              />
            </div>
            <div className="flex justify-between font-mono text-[10px] text-neutral-subtext uppercase tracking-widest">
              <span>Threshold: {Math.round(alert.threshold * 100)}%</span>
              <span>{alert.triggered ? 'Triggered' : 'Below threshold'}</span>
            </div>
          </div>

          {/* Metadata grid */}
          <div className="grid grid-cols-2 gap-3">
            {[
              { label: 'Model', value: alert.model_type },
              { label: 'Schema', value: alert.feature_schema ?? '—' },
              {
                label: 'Detected',
                value: new Date(alert.timestamp).toLocaleString(),
              },
              { label: 'Window ID', value: alert.window_id },
            ].map(({ label, value }) => (
              <div key={label} className="disclosure-item">
                <p className="text-quiet mb-1">{label}</p>
                <p className="font-mono text-xs text-neutral-white break-all">{value}</p>
              </div>
            ))}
          </div>

          {/* Network flow details */}
          {explainParts.length > 0 && (
            <div>
              <p className="text-quiet mb-3">Network Flow Details</p>
              <div className="sentinel-terminal space-y-1">
                {explainParts.map(({ key, value }, i) => (
                  <div key={i} className="flex gap-2">
                    <span className="text-sentinel-accent w-28 shrink-0">{key}</span>
                    <span className="text-neutral-white">{value || '—'}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Investigations */}
          <div>
            <p className="text-quiet mb-3">Investigation History</p>
            {loadingInv ? (
              <p className="font-mono text-xs text-neutral-subtext">Loading...</p>
            ) : investigations.length === 0 ? (
              <p className="font-mono text-xs text-neutral-subtext">No investigations yet.</p>
            ) : (
              <div className="space-y-2">
                {investigations.map((inv) => (
                  <div key={inv.id} className="disclosure-item space-y-1">
                    <div className="flex items-center justify-between">
                      <span
                        className={`inline-flex items-center rounded-full border px-2 py-0.5 font-mono text-[10px] font-semibold uppercase tracking-[0.18em] ${STATUS_COLORS[inv.status] ?? 'text-neutral-subtext'}`}
                      >
                        {STATUS_LABELS[inv.status] ?? inv.status}
                      </span>
                      <span className="font-mono text-[10px] text-neutral-subtext">
                        {inv.user_id ?? 'unknown'} · {new Date(inv.updated_at).toLocaleString()}
                      </span>
                    </div>
                    {inv.notes && (
                      <p className="font-mono text-xs text-neutral-white whitespace-pre-wrap">{inv.notes}</p>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Notes input */}
          <div className="space-y-2">
            <label htmlFor="inv-notes" className="text-quiet">
              Notes (optional)
            </label>
            <textarea
              id="inv-notes"
              rows={3}
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="Add context or findings..."
              className="w-full rounded-md border border-sentinel-border/40 bg-sentinel-elevated/40 px-3 py-2 font-mono text-xs text-neutral-white placeholder:text-neutral-subtext focus:border-sentinel-accent/60 focus:outline-none resize-none"
            />
          </div>

          {error && (
            <p className="font-mono text-xs text-threat-muted">{error}</p>
          )}

          {/* Action buttons */}
          <div className="flex flex-wrap gap-2">
            {currentStatus !== 'investigating' && (
              <button
                onClick={() => handleAction('investigating')}
                disabled={submitting}
                className="btn-quiet hover:border-anomaly-accent/40 hover:text-anomaly-light disabled:opacity-50"
              >
                Start Investigation
              </button>
            )}
            {currentStatus !== 'resolved' && (
              <button
                onClick={() => handleAction('resolved')}
                disabled={submitting}
                className="btn-quiet hover:border-insight-blue/40 hover:text-insight-light disabled:opacity-50"
              >
                Mark Resolved
              </button>
            )}
            {currentStatus !== 'false_positive' && (
              <button
                onClick={() => handleAction('false_positive')}
                disabled={submitting}
                className="btn-quiet hover:border-sentinel-border/60 hover:text-neutral-subtext disabled:opacity-50"
              >
                Mark False Positive
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  )
}

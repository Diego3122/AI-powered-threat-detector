import { useEffect, useState } from 'react'
import { ChevronDown } from 'lucide-react'

import { threatAPI } from '@/api/client'
import IncidentList from '@/components/IncidentList'
import ModelConfidence from '@/components/ModelConfidence'
import ThreatTimeline from '@/components/ThreatTimeline'

interface AlertRecord {
  model_score: number
}

interface ModelInfo {
  accuracy?: number
}

export default function Overview() {
  const [expandedSection, setExpandedSection] = useState<string | null>(null)
  const [summary, setSummary] = useState({
    alertsRecorded: 0,
    highSeverityAlerts: 0,
    modelConfidence: null as number | null,
  })

  useEffect(() => {
    let active = true

    const loadSummary = async () => {
      try {
        const [alerts, model] = await Promise.all([
          threatAPI.getThreatTimeline(),
          threatAPI.getModelPerformance(),
        ])

        if (!active) return

        const alertList = Array.isArray(alerts) ? (alerts as AlertRecord[]) : []
        const modelInfo = (model || null) as ModelInfo | null

        setSummary({
          alertsRecorded: alertList.length,
          highSeverityAlerts: alertList.filter((alert) => alert.model_score >= 0.8).length,
          modelConfidence: modelInfo?.accuracy ? Math.round(modelInfo.accuracy * 100) : null,
        })
      } catch (error) {
        console.error('Failed to load overview summary:', error)
      }
    }

    loadSummary()
    const intervalId = window.setInterval(loadSummary, 10000)
    return () => {
      active = false
      window.clearInterval(intervalId)
    }
  }, [])

  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <p className="sentinel-kicker">Threat analysis / overview</p>
        <div className="flex flex-col gap-6 xl:flex-row xl:items-end xl:justify-between">
          <div className="space-y-3">
            <h1 className="heading-editorial">Threat Posture</h1>
            <p className="max-w-2xl text-sm text-neutral-subtext">
              Live network telemetry, incident severity, and model confidence rendered through a darker operator-style console.
            </p>
          </div>

          <div className="sentinel-card-soft px-5 py-4">
            <p className="text-quiet">Current Date</p>
            <p className="mt-2 text-sm text-neutral-white">
              {new Date().toLocaleDateString('en-US', {
                weekday: 'long',
                month: 'long',
                day: 'numeric',
                year: 'numeric',
              })}
            </p>
          </div>
        </div>
      </section>

      <section className="grid gap-4 md:grid-cols-3">
        <div className="sentinel-card p-5">
          <p className="text-quiet">Alerts Recorded</p>
          <div className="mt-4 flex items-end justify-between">
            <span className="font-editorial text-4xl font-bold text-neutral-white">{summary.alertsRecorded}</span>
            <span className="threat-badge">Live Feed</span>
          </div>
        </div>

        <div className="sentinel-card p-5">
          <p className="text-quiet">High Severity</p>
          <div className="mt-4 flex items-end justify-between">
            <span className="font-editorial text-4xl font-bold text-threat-muted">{summary.highSeverityAlerts}</span>
            <span className="anomaly-badge">Priority</span>
          </div>
        </div>

        <div className="sentinel-card p-5">
          <p className="text-quiet">Model Confidence</p>
          <div className="mt-4 flex items-end justify-between">
            <span className="font-editorial text-4xl font-bold text-sentinel-accent">
              {summary.modelConfidence !== null ? `${summary.modelConfidence}%` : 'N/A'}
            </span>
            <span className="insight-badge">AI Signal</span>
          </div>
        </div>
      </section>

      <section className="grid gap-6 xl:grid-cols-[1.85fr_1fr]">
        <div className="space-y-4">
          <div className="viz-hero">
            <ThreatTimeline />
          </div>

          <button
            onClick={() => setExpandedSection(expandedSection === 'timeline' ? null : 'timeline')}
            className="btn-quiet"
          >
            <span className={`transition-transform ${expandedSection === 'timeline' ? 'rotate-180' : ''}`}>
              <ChevronDown size={14} />
            </span>
            {expandedSection === 'timeline' ? 'Hide Timeline Intel' : 'Show Timeline Intel'}
          </button>

          {expandedSection === 'timeline' && (
            <div className="disclosure-item grid gap-4 md:grid-cols-3">
              <div className="sentinel-card-soft p-4">
                <p className="text-quiet">Total Events</p>
                <p className="mt-3 font-editorial text-3xl font-bold text-neutral-white">{summary.alertsRecorded}</p>
              </div>
              <div className="sentinel-card-soft p-4">
                <p className="text-quiet">Escalations</p>
                <p className="mt-3 font-editorial text-3xl font-bold text-threat-muted">{summary.highSeverityAlerts}</p>
              </div>
              <div className="sentinel-card-soft p-4">
                <p className="text-quiet">Confidence</p>
                <p className="mt-3 font-editorial text-3xl font-bold text-sentinel-accent">
                  {summary.modelConfidence !== null ? `${summary.modelConfidence}%` : 'N/A'}
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="space-y-6">
          <section className="card-minimal space-y-5">
            <div className="space-y-2">
              <p className="sentinel-kicker">AI performance</p>
              <h2 className="heading-medium">Model Confidence</h2>
            </div>

            <ModelConfidence />

            <button
              onClick={() => setExpandedSection(expandedSection === 'model' ? null : 'model')}
              className="btn-quiet"
            >
              <span className={`transition-transform ${expandedSection === 'model' ? 'rotate-180' : ''}`}>
                <ChevronDown size={14} />
              </span>
              {expandedSection === 'model' ? 'Hide Model Intel' : 'Show Model Intel'}
            </button>

            {expandedSection === 'model' && (
              <div className="disclosure-item space-y-3 text-sm">
                <div className="flex justify-between gap-4">
                  <span className="text-neutral-subtext">Active model confidence</span>
                  <span className="font-medium text-neutral-white">
                    {summary.modelConfidence !== null ? `${summary.modelConfidence}%` : 'N/A'}
                  </span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-neutral-subtext">Alerts recorded</span>
                  <span className="font-medium text-neutral-white">{summary.alertsRecorded}</span>
                </div>
                <div className="flex justify-between gap-4">
                  <span className="text-neutral-subtext">High-severity alerts</span>
                  <span className="font-medium text-neutral-white">{summary.highSeverityAlerts}</span>
                </div>
              </div>
            )}
          </section>
        </div>
      </section>

      <section className="card-minimal space-y-5">
        <div className="space-y-2">
          <p className="sentinel-kicker">Operator queue</p>
          <h2 className="heading-medium">Active Incidents</h2>
        </div>

        <IncidentList />

        <div className="flex flex-wrap gap-3 pt-2">
          <button className="btn-quiet">View All Incidents</button>
          <button className="btn-quiet">Generate Report</button>
          <button className="btn-quiet">Export Data</button>
        </div>
      </section>
    </div>
  )
}

import { useEffect, useState } from 'react'
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'

import { threatAPI } from '@/api/client'

interface AlertRecord {
  id: number
  timestamp: number
  model_score: number
  triggered: boolean
}

interface TimelineBucket {
  time: string
  events: number
  anomalies: number
}

const HOURS_TO_SHOW = 24

const createBuckets = () => {
  const now = new Date()
  now.setMinutes(0, 0, 0)

  return Array.from({ length: HOURS_TO_SHOW }, (_, index) => {
    const bucketTime = new Date(now)
    bucketTime.setHours(now.getHours() - (HOURS_TO_SHOW - 1 - index))
    return {
      key: bucketTime.getTime(),
      time: bucketTime.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      events: 0,
      anomalies: 0,
    }
  })
}

const buildTimeline = (alerts: AlertRecord[]): TimelineBucket[] => {
  const buckets = createBuckets()
  const bucketIndex = new Map(buckets.map((bucket, index) => [bucket.key, index]))

  alerts.forEach((alert) => {
    const timestamp = new Date(alert.timestamp)
    timestamp.setMinutes(0, 0, 0)
    const key = timestamp.getTime()
    const index = bucketIndex.get(key)
    if (index === undefined) return

    buckets[index].events += 1
    if (alert.triggered || alert.model_score >= 0.8) {
      buckets[index].anomalies += 1
    }
  })

  return buckets.map(({ time, events, anomalies }) => ({ time, events, anomalies }))
}

export default function ThreatTimeline() {
  const [data, setData] = useState<TimelineBucket[]>(createBuckets())
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true

    const loadTimeline = async () => {
      try {
        const alerts = await threatAPI.getThreatTimeline()
        if (!active) return
        setData(buildTimeline(Array.isArray(alerts) ? alerts : []))
      } catch (error) {
        console.error('Failed to build threat timeline:', error)
      } finally {
        if (active) setLoading(false)
      }
    }

    loadTimeline()
    const intervalId = window.setInterval(loadTimeline, 10000)
    return () => {
      active = false
      window.clearInterval(intervalId)
    }
  }, [])

  return (
    <div className="flex h-full w-full flex-col p-6 md:p-8">
      <div className="mb-6 flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
        <div className="space-y-2">
          <p className="sentinel-kicker">Live threat feed</p>
          <h3 className="font-editorial text-2xl font-bold uppercase tracking-tight text-neutral-white">
            24-Hour Activity
          </h3>
          <p className="text-sm text-neutral-subtext">
            {loading ? 'Loading live detection data...' : 'Detection events and elevated alerts across the last 24 hours'}
          </p>
        </div>

        <div className="rounded-xl border border-sentinel-accent/20 bg-sentinel-accentStrong/10 px-4 py-3">
          <p className="font-mono text-[10px] uppercase tracking-[0.2em] text-sentinel-accent">Telemetry</p>
          <p className="mt-1 text-sm text-neutral-white">Streaming in 10s intervals</p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 5, right: 30, left: 0, bottom: 50 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#24314e" />
          <XAxis
            dataKey="time"
            stroke="#a3aac4"
            tick={{ fill: '#a3aac4', fontSize: 11 }}
            label={{ value: 'Time', position: 'insideBottomRight', offset: -10, fill: '#a3aac4' }}
          />
          <YAxis
            stroke="#a3aac4"
            tick={{ fill: '#a3aac4', fontSize: 11 }}
            label={{ value: 'Count', angle: -90, position: 'insideLeft', fill: '#a3aac4' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#0f1930',
              border: '1px solid #40485d',
              borderRadius: 12,
              color: '#dee5ff',
            }}
            labelStyle={{ color: '#dee5ff', fontSize: 11, textTransform: 'uppercase', letterSpacing: '0.16em' }}
          />
          <Legend
            wrapperStyle={{ paddingTop: '20px', fontSize: '11px', letterSpacing: '0.12em', textTransform: 'uppercase' }}
            iconType="line"
          />
          <Line type="monotone" dataKey="events" stroke="#7ee5ff" dot={false} strokeWidth={2.5} name="Alerts Recorded" />
          <Line
            type="monotone"
            dataKey="anomalies"
            stroke="#ff6f7e"
            dot={false}
            strokeWidth={2.5}
            name="High-Severity Alerts"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}

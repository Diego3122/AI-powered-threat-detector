import { useEffect, useState } from 'react'

import { threatAPI } from '@/api/client'

interface Model {
  id: number
  model_type: string
  version?: string
  accuracy?: number
  f1_score?: number
  roc_auc?: number
  active: boolean
}

export default function ModelConfidence() {
  const [model, setModel] = useState<Model | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchModel = async () => {
      try {
        const data = await threatAPI.getModelPerformance()
        if (data) {
          setModel(data)
        }
      } catch (error) {
        console.error('Failed to load model:', error)
      } finally {
        setLoading(false)
      }
    }

    fetchModel()
  }, [])

  if (loading) {
    return <div className="text-sm text-neutral-subtext">Loading model data...</div>
  }

  if (!model) {
    return <div className="text-sm text-neutral-subtext">No active model available</div>
  }

  const confidence = model.accuracy ? Math.round(model.accuracy * 100) : null

  return (
    <div className="space-y-6">
      <div className="rounded-xl border border-sentinel-accent/20 bg-sentinel-accentStrong/10 p-6 shadow-glow">
        <p className="text-quiet">Operational Confidence</p>
        <div className="mt-3 flex items-end gap-4">
          <div className="font-editorial text-6xl font-bold tracking-tight text-sentinel-accent">
            {confidence !== null ? `${confidence}%` : 'N/A'}
          </div>
          <div className="pb-2 text-sm text-neutral-text">Model Confidence</div>
        </div>
      </div>

      <div className="grid gap-3 md:grid-cols-2">
        <div className="sentinel-card-soft p-4">
          <p className="text-quiet">F1 Score</p>
          <p className="mt-3 font-editorial text-3xl font-bold text-anomaly-light">
            {model.f1_score ? `${(model.f1_score * 100).toFixed(1)}%` : 'N/A'}
          </p>
        </div>
        <div className="sentinel-card-soft p-4">
          <p className="text-quiet">ROC AUC</p>
          <p className="mt-3 font-editorial text-3xl font-bold text-sentinel-success">
            {model.roc_auc ? `${(model.roc_auc * 100).toFixed(1)}%` : 'N/A'}
          </p>
        </div>
      </div>

      <div className="space-y-3 border-t border-sentinel-border/20 pt-4">
        <div className="flex items-center justify-between">
          <span className="text-sm text-neutral-subtext">Active Model</span>
          <span className="font-medium text-neutral-white">
            {model.model_type} {model.version ? `v${model.version}` : ''}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-sm text-neutral-subtext">Status</span>
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-sentinel-accent animate-pulse" />
            <span className="text-sm font-medium text-sentinel-accent">Operational</span>
          </div>
        </div>
      </div>
    </div>
  )
}

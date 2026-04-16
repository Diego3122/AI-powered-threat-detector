import { useEffect, useState } from 'react'
import { threatAPI } from '@/api/client'

interface ModelInfo {
  id: number
  model_type: string
  version?: string
  accuracy?: number
  f1_score?: number
  roc_auc?: number
  active: boolean
}

export default function Models() {
  const [model, setModel] = useState<ModelInfo | null>(null)
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

  return (
    <div className="space-y-8">
      <div className="space-y-3">
        <p className="sentinel-kicker">Machine intelligence / diagnostics</p>
        <h1 className="heading-editorial">Models</h1>
        <p className="max-w-2xl text-sm text-neutral-subtext">
          Performance, status, and versioning for the active detection model without changing any of the existing backend behavior.
        </p>
      </div>

      {loading ? (
        <div className="sentinel-card p-10 text-center text-neutral-subtext">
          <p>Loading model data...</p>
        </div>
      ) : !model ? (
        <div className="sentinel-card p-10 text-center text-neutral-subtext">
          <p>No active model found</p>
        </div>
      ) : (
        <div className="space-y-6">
          <div className="sentinel-card p-8">
            <div className="space-y-6">
              <div>
                <p className="sentinel-kicker">Deployment status</p>
                <h2 className="mt-2 font-editorial text-3xl font-bold uppercase tracking-tight text-neutral-white">
                  Active Model
                </h2>
                <div className="flex items-center gap-3">
                  <div className="h-3 w-3 rounded-full bg-sentinel-accent animate-pulse" />
                  <span className="font-medium text-sentinel-accent">Operational</span>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-6">
                <div>
                  <div className="text-quiet mb-2">Model Type</div>
                  <div className="text-lg font-medium text-neutral-white">{model.model_type}</div>
                </div>
                <div>
                  <div className="text-quiet mb-2">Version</div>
                  <div className="text-lg font-medium text-neutral-white">{model.version || 'N/A'}</div>
                </div>
              </div>
            </div>
          </div>

          <div className="space-y-4">
            <h3 className="heading-medium">Performance Metrics</h3>

            <div className="grid grid-cols-3 gap-4">
              <div className="sentinel-card-soft p-6">
                <div className="text-quiet mb-3">Accuracy</div>
                <div className="font-editorial text-4xl font-bold text-sentinel-accent">
                  {model.accuracy ? Math.round(model.accuracy * 100) : 'N/A'}%
                </div>
              </div>

              <div className="sentinel-card-soft p-6">
                <div className="text-quiet mb-3">F1 Score</div>
                <div className="font-editorial text-4xl font-bold text-anomaly-light">
                  {model.f1_score ? (model.f1_score * 100).toFixed(1) : 'N/A'}%
                </div>
              </div>

              <div className="sentinel-card-soft p-6">
                <div className="text-quiet mb-3">ROC AUC</div>
                <div className="font-editorial text-4xl font-bold text-threat-muted">
                  {model.roc_auc ? (model.roc_auc * 100).toFixed(1) : 'N/A'}%
                </div>
              </div>
            </div>
          </div>

          <div className="sentinel-card space-y-4 p-6">
            <h3 className="heading-medium">Model Details</h3>

            <div className="space-y-3">
              <div className="flex items-center justify-between border-b border-sentinel-border/20 pb-3">
                <span className="text-neutral-subtext">Model Type</span>
                <span className="font-medium text-neutral-white">{model.model_type}</span>
              </div>
              <div className="flex items-center justify-between border-b border-sentinel-border/20 pb-3">
                <span className="text-neutral-subtext">Version</span>
                <span className="font-medium text-neutral-white">{model.version || 'Latest'}</span>
              </div>
              <div className="flex items-center justify-between border-b border-sentinel-border/20 pb-3">
                <span className="text-neutral-subtext">Status</span>
                <span className="font-medium text-sentinel-accent">Active</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-neutral-subtext">ID</span>
                <span className="font-medium text-neutral-white">{model.id}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

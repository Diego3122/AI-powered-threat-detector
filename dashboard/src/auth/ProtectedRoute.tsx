import { Navigate, Outlet, useLocation } from 'react-router-dom'

import { useAuth } from '@/auth/AuthProvider'

export default function ProtectedRoute() {
  const { user, loading } = useAuth()
  const location = useLocation()

  if (loading) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-sentinel-bg text-neutral-white">
        <div className="sentinel-card px-8 py-6 text-center">
          <p className="sentinel-kicker">Authorizing session</p>
          <h1 className="mt-3 font-editorial text-2xl font-bold uppercase tracking-tight">Threat Detector</h1>
        </div>
      </div>
    )
  }

  if (!user) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }

  return <Outlet />
}

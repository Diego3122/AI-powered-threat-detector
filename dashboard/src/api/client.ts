import axios from 'axios'

const TOKEN_STORAGE_KEY = 'threat_detector_token'

const inferApiBase = () => {
  const configured = import.meta.env.VITE_API_URL?.trim()
  if (configured) return configured.replace(/\/+$/, '')
  if (typeof window !== 'undefined') {
    return window.location.origin
  }
  return 'http://localhost:8000'
}

const API_BASE = inferApiBase()

export interface AuthenticatedUser {
  username: string
  roles: string[]
}

export interface TokenResponse {
  access_token: string
  token_type: string
}

export const getStoredToken = () => {
  if (typeof window === 'undefined') return null
  return window.sessionStorage.getItem(TOKEN_STORAGE_KEY)
}

export const setStoredToken = (token: string) => {
  if (typeof window === 'undefined') return
  window.sessionStorage.setItem(TOKEN_STORAGE_KEY, token)
  window.localStorage.removeItem(TOKEN_STORAGE_KEY)
}

export const clearStoredToken = () => {
  if (typeof window === 'undefined') return
  window.sessionStorage.removeItem(TOKEN_STORAGE_KEY)
}

export const apiClient = axios.create({
  baseURL: API_BASE,
})

apiClient.interceptors.request.use((config) => {
  const token = getStoredToken()
  if (token) {
    config.headers = config.headers ?? {}
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

export const authAPI = {
  login: async (username: string, password: string): Promise<TokenResponse> => {
    const response = await apiClient.post('/api/auth/login', { username, password })
    return response.data
  },
  me: async (): Promise<AuthenticatedUser> => {
    const response = await apiClient.get('/api/auth/me')
    return response.data
  },
}

export const threatAPI = {
  // Get recent threat timeline data
  getThreatTimeline: async () => {
    try {
      const response = await apiClient.get('/api/alerts?limit=100')
      return response.data
    } catch (error) {
      console.error('Failed to fetch threat timeline:', error)
      return null
    }
  },

  // Get active incidents
  getIncidents: async () => {
    try {
      const response = await apiClient.get('/api/alerts?triggered=true')
      return response.data
    } catch (error) {
      console.error('Failed to fetch incidents:', error)
      return null
    }
  },

  // Get model performance
  getModelPerformance: async () => {
    try {
      // Try the active endpoint first, fallback to list
      const response = await apiClient.get('/api/models/active')
      return response.data || null
    } catch (error) {
      // Fallback to list endpoint
      try {
        const response = await apiClient.get('/api/models?active=true')
        return response.data?.[0] || null
      } catch (fallbackError) {
        console.error('Failed to fetch model performance:', fallbackError)
        return null
      }
    }
  },

  // Get health check
  getHealth: async () => {
    try {
      const response = await apiClient.get('/health')
      return response.data
    } catch (error) {
      console.error('API health check failed:', error)
      return null
    }
  },

  // Get audit logs
  getAuditLogs: async (limit = 100, offset = 0) => {
    try {
      const response = await apiClient.get(`/api/audit/logs?limit=${limit}&offset=${offset}`)
      return response.data
    } catch (error) {
      console.error('Failed to fetch audit logs:', error)
      return []
    }
  },

  // Get investigations
  getInvestigations: async (status?: string, limit = 100, offset = 0) => {
    try {
      const params = new URLSearchParams({ limit: limit.toString(), offset: offset.toString() })
      if (status) params.append('status', status)
      const response = await apiClient.get(`/api/investigations?${params}`)
      return response.data
    } catch (error) {
      console.error('Failed to fetch investigations:', error)
      return []
    }
  },

  // Get a single alert by id
  getAlert: async (alertId: number) => {
    const response = await apiClient.get(`/api/alerts/${alertId}`)
    return response.data
  },

  // Get investigations for a specific alert
  getAlertInvestigations: async (alertId: number) => {
    const response = await apiClient.get(`/api/alerts/${alertId}/investigations`)
    return response.data
  },

  // Create an investigation for an alert
  createInvestigation: async (
    alertId: number,
    data: { status: string; notes?: string },
  ) => {
    const response = await apiClient.post(`/api/alerts/${alertId}/investigations`, data)
    return response.data
  },

  // Update an existing investigation
  updateInvestigation: async (
    investigationId: number,
    data: { status?: string; notes?: string },
  ) => {
    const response = await apiClient.put(`/api/investigations/${investigationId}`, data)
    return response.data
  },
}

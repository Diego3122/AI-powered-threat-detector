import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from 'react'

import {
  authAPI,
  clearStoredToken,
  getStoredToken,
  setStoredToken,
  type AuthenticatedUser,
} from '@/api/client'

interface AuthContextValue {
  user: AuthenticatedUser | null
  loading: boolean
  login: (username: string, password: string) => Promise<void>
  logout: () => void
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined)

export function AuthProvider({ children }: { children: ReactNode }) {
  const [token, setToken] = useState<string | null>(() => getStoredToken())
  const [user, setUser] = useState<AuthenticatedUser | null>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let active = true

    const restoreSession = async () => {
      if (!token) {
        if (active) {
          setUser(null)
          setLoading(false)
        }
        return
      }

      try {
        const currentUser = await authAPI.me()
        if (active) {
          setUser(currentUser)
        }
      } catch {
        clearStoredToken()
        if (active) {
          setToken(null)
          setUser(null)
        }
      } finally {
        if (active) {
          setLoading(false)
        }
      }
    }

    setLoading(true)
    restoreSession()

    return () => {
      active = false
    }
  }, [token])

  const login = useCallback(async (username: string, password: string) => {
    const tokenResponse = await authAPI.login(username, password)
    setStoredToken(tokenResponse.access_token)
    setToken(tokenResponse.access_token)
    const currentUser = await authAPI.me()
    setUser(currentUser)
  }, [])

  const logout = useCallback(() => {
    clearStoredToken()
    setToken(null)
    setUser(null)
  }, [])

  const value = useMemo(
    () => ({
      user,
      loading,
      login,
      logout,
    }),
    [user, loading, login, logout]
  )

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

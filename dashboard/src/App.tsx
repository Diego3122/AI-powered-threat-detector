import '@/styles/globals.css'
import { BrowserRouter as Router, Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider } from '@/auth/AuthProvider'
import ProtectedRoute from '@/auth/ProtectedRoute'
import Layout from '@/components/Layout'
import Overview from '@/pages/Overview'
import Incidents from '@/pages/Incidents'
import Models from '@/pages/Models'
import Logs from '@/pages/Logs'
import Login from '@/pages/Login'

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<Login />} />
          <Route element={<ProtectedRoute />}>
            <Route element={<Layout />}>
              <Route path="/" element={<Overview />} />
              <Route path="/incidents" element={<Incidents />} />
              <Route path="/models" element={<Models />} />
              <Route path="/logs" element={<Logs />} />
            </Route>
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>
    </AuthProvider>
  )
}

export default App

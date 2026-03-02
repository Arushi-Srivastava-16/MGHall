import React, { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom'
import { Home, MessageSquare, Search, Info, Database, Activity, FileText } from 'lucide-react'
import HomePage from './pages/HomePage'
import ModelsPage from './pages/ModelsPage'
import PatternsPage from './pages/PatternsPage'
import ResultsPage from './pages/ResultsPage'
import ChatDemoPage from './pages/ChatDemoPage'
import ForensicPage from './pages/ForensicPage'
import './App.css'

function Navbar() {
  const location = useLocation()
  const [showInfoDropdown, setShowInfoDropdown] = useState(false)

  const isActive = (path) => location.pathname === path

  return (
    <nav className="navbar">
      <div className="nav-container">
        <Link to="/" className="nav-logo">
          <div className="logo-icon">LLMShield</div>
          <span>Framework</span>
        </Link>
        <div className="nav-links">
          <Link to="/" className={`nav-item ${isActive('/') ? 'active' : ''}`}>
            <Home size={18} />
            <span>Home</span>
          </Link>

          <Link to="/chat" className={`nav-item ${isActive('/chat') ? 'active' : ''}`}>
            <MessageSquare size={18} />
            <span>Live Demo</span>
          </Link>

          <Link to="/forensic" className={`nav-item ${isActive('/forensic') ? 'active' : ''}`}>
            <Search size={18} />
            <span>Forensic</span>
          </Link>

          {/* Info Dropdown */}
          <div
            className="nav-dropdown-container"
            onMouseEnter={() => setShowInfoDropdown(true)}
            onMouseLeave={() => setShowInfoDropdown(false)}
          >
            <div className={`nav-item ${isActive('/models') || isActive('/results') ? 'active' : ''}`}>
              <Info size={18} />
              <span>Info</span>
            </div>

            {showInfoDropdown && (
              <div className="nav-dropdown-menu">
                <Link to="/models" className="dropdown-item">
                  <Database size={16} /> Models
                </Link>
                <Link to="/results" className="dropdown-item">
                  <Activity size={16} /> Results
                </Link>
                <Link to="/patterns" className="dropdown-item">
                  <FileText size={16} /> Patterns
                </Link>
              </div>
            )}
          </div>
        </div>
      </div>
    </nav>
  )
}

function App() {
  return (
    <Router>
      <div className="app">
        <Navbar />
        <main className="main-content">
          <Routes>
            <Route path="/chat" element={<ChatDemoPage />} />
            <Route path="/forensic" element={<ForensicPage />} />
            <Route path="/" element={<HomePage />} />
            <Route path="/models" element={<ModelsPage />} />
            <Route path="/results" element={<ResultsPage />} />
            <Route path="/patterns" element={<PatternsPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App


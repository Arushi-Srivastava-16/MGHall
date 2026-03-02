import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import axios from 'axios'
import { ArrowRight, ShieldCheck, Activity, BrainCircuit, Search, Database, FileText, TrendingUp, Clock, AlertCircle } from 'lucide-react'
import ArchitectureDiagram from '../components/ArchitectureDiagram'
import './HomePage.css'

function HomePage() {
  const [stats, setStats] = useState(null)

  useEffect(() => {
    axios.get('http://localhost:8000/api/stats')
      .then(res => setStats(res.data))
      .catch(err => console.error("Stats fetch error:", err))
  }, [])

  return (
    <div className="home-container">
      {/* Hero Section */}
      <section className="hero-section">
        <div className="hero-content">
          <div className="hero-badge" style={{ display: 'none' }}></div>
          <h1>LLMShield Framework</h1>
          <p className="hero-subtitle">
            A neurosymbolic framework for real-time hallucination detection, explanation, and correction in Large Language Models.
          </p>
          <div className="hero-actions">
            <Link to="/chat" className="btn-primary-lg">
              Launch Live Demo <ArrowRight size={20} />
            </Link>
            <Link to="/forensic" className="btn-secondary-lg">
              Forensic Analysis
            </Link>
          </div>
        </div>

        {/* Enhanced Stats Dashboard */}
        {stats && (
          <div className="stats-dashboard">
            <div className="stats-grid">
              {/* Primary Stats - Larger Cards */}
              <div className="stat-card primary">
                <div className="stat-icon blue">
                  <Database size={24} />
                </div>
                <div className="stat-content">
                  <div className="stat-value">{stats.models_loaded}</div>
                  <div className="stat-label">Active Models</div>
                </div>
              </div>

              <div className="stat-card primary">
                <div className="stat-icon green">
                  <ShieldCheck size={24} />
                </div>
                <div className="stat-content">
                  <div className="stat-value">{(stats.avg_safety_score * 100).toFixed(0)}%</div>
                  <div className="stat-label">Safety Score</div>
                </div>
              </div>

              <div className="stat-card primary">
                <div className="stat-icon purple">
                  <Activity size={24} />
                </div>
                <div className="stat-content">
                  <div className="stat-value text-green">Online</div>
                  <div className="stat-label">System Status</div>
                  <div className="stat-sub">{stats.uptime} uptime</div>
                </div>
              </div>

              {/* Secondary Stats - Smaller Cards */}
              <div className="stat-card secondary">
                <div className="stat-mini-icon">
                  <TrendingUp size={18} color="#3b82f6" />
                </div>
                <div className="stat-mini-content">
                  <div className="stat-mini-value">{stats.total_queries}</div>
                  <div className="stat-mini-label">Total Queries</div>
                </div>
              </div>

              <div className="stat-card secondary">
                <div className="stat-mini-icon">
                  <Clock size={18} color="#10b981" />
                </div>
                <div className="stat-mini-content">
                  <div className="stat-mini-value">{stats.avg_response_time}</div>
                  <div className="stat-mini-label">Avg Response</div>
                </div>
              </div>

              <div className="stat-card secondary">
                <div className="stat-mini-icon">
                  <AlertCircle size={18} color="#8b5cf6" />
                </div>
                <div className="stat-mini-content">
                  <div className="stat-mini-value">{stats.hallucinations_prevented}</div>
                  <div className="stat-mini-label">Issues Prevented</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Architecture Diagram */}
      <ArchitectureDiagram />

      {/* Capabilities Grid */}
      <section className="features-section">
        <h2 className="section-title">Core Capabilities</h2>
        <div className="features-grid-lg">
          <div className="feature-card-lg">
            <div className="icon-wrapper blue"><Activity size={24} /></div>
            <h3>Real-Time Risk Scoring</h3>
            <p>VulnerabilityGAT measures hallucination risk live as the model generates reasoning steps.</p>
          </div>
          <div className="feature-card-lg">
            <div className="icon-wrapper purple"><BrainCircuit size={24} /></div>
            <h3>Dynamic Branching</h3>
            <p>Visualizes potential reasoning failures and alternative paths via candidate sampling.</p>
          </div>
          <div className="feature-card-lg">
            <div className="icon-wrapper green"><ShieldCheck size={24} /></div>
            <h3>RAG Intervention</h3>
            <p>Logic-aware retrieval injection corrects hallucinations before they reach the user.</p>
          </div>
          <div className="feature-card-lg">
            <div className="icon-wrapper orange"><Search size={24} /></div>
            <h3>Forensic Fingerprinting</h3>
            <p>Identifies source models and stylistic signatures (e.g., Llama vs GPT-4).</p>
          </div>
        </div>
      </section>

      {/* Info Links */}
      <section className="info-section">
        <h2 className="section-title">System Information</h2>
        <div className="info-grid">
          <Link to="/models" className="info-card">
            <Database size={20} />
            <div>
              <h4>Model Registry</h4>
              <span>View loaded GNN architectures</span>
            </div>
          </Link>
          <Link to="/results" className="info-card">
            <Activity size={20} />
            <div>
              <h4>Training Results</h4>
              <span>Experiment metrics & history</span>
            </div>
          </Link>
          <Link to="/patterns" className="info-card">
            <FileText size={20} />
            <div>
              <h4>Pattern Database</h4>
              <span>Known hallucination types</span>
            </div>
          </Link>
        </div>
      </section>
    </div>
  )
}

export default HomePage


import React, { useState, useEffect } from 'react'
import axios from 'axios'
import './SharedStyles.css'

function PatternsPage() {
  const [patterns, setPatterns] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get('http://localhost:8000/api/patterns')
      .then(res => {
        setPatterns(res.data)
        setLoading(false)
      })
      .catch(err => {
        console.error("Pattern fetch failed", err)
        setLoading(false)
      })
  }, [])

  return (
    <div className="page">
      <h1>Hallucination Pattern Database</h1>
      <div className="card">
        <p>This database catalogs the specific types of inconsistencies detected by the framework's Graph Attention Networks.</p>
        <p>Each pattern represents a distinct structural anomaly in the reasoning graph.</p>
      </div>

      {loading ? <p>Loading patterns...</p> : (
        <div style={{ display: 'grid', gap: '20px' }}>
          {patterns.map(pat => (
            <div key={pat.id} className="card" style={{ borderLeft: `5px solid ${pat.severity === 'Critical' ? '#ef4444' : pat.severity === 'High' ? '#f59e0b' : '#3b82f6'}` }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '10px' }}>
                <h3 style={{ margin: 0 }}>{pat.name}</h3>
                <span style={{
                  background: pat.severity === 'Critical' ? '#fee2e2' : pat.severity === 'High' ? '#fef3c7' : '#dbeafe',
                  color: pat.severity === 'Critical' ? '#b91c1c' : pat.severity === 'High' ? '#b45309' : '#1e40af',
                  padding: '4px 8px', borderRadius: '4px', fontSize: '0.8rem', fontWeight: 'bold'
                }}>{pat.severity}</span>
              </div>
              <p style={{ marginBottom: '16px' }}>{pat.description}</p>
              <div style={{ padding: '8px', background: '#f8fafc', borderRadius: '6px', fontSize: '0.9rem', color: '#64748b', border: '1px solid #e2e8f0' }}>
                <strong>Detection Method:</strong> {pat.detection_method}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default PatternsPage

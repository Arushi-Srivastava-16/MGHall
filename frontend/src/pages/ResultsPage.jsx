import React, { useState, useEffect } from 'react'
import axios from 'axios'
import './SharedStyles.css'

function ResultsPage() {
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get('http://localhost:8000/api/results')
      .then(res => {
        setResults(res.data)
        setLoading(false)
      })
      .catch(err => {
        console.error("Results fetch failed", err)
        setLoading(false)
      })
  }, [])

  return (
    <div className="page">
      <h1>Validation Results</h1>
      <div className="card">
        <p>Performance metrics from the latest Training and Evaluation runs for each domain.</p>
      </div>

      {loading ? (
        <p>Loading results...</p>
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
          {results.map((exp) => (
            <div key={exp.experiment_id} className="card" style={{ margin: 0, display: 'flex', gap: '24px', alignItems: 'center' }}>
              <div style={{ minWidth: '200px' }}>
                <h3 style={{ margin: '0 0 4px 0' }}>{exp.domain}</h3>
                <span style={{ fontSize: '0.8rem', color: '#94a3b8', fontFamily: 'JetBrains Mono' }}>{exp.experiment_id}</span>
              </div>

              <div style={{ flex: 1, display: 'flex', gap: '32px' }}>
                {Object.entries(exp.metrics).map(([key, value]) => (
                  <div key={key}>
                    <div style={{ fontSize: '0.75rem', textTransform: 'uppercase', color: '#64748b', fontWeight: '600' }}>
                      {key.replace('_', ' ')}
                    </div>
                    <div style={{ fontSize: '1.5rem', fontWeight: '700', color: '#0f172a' }}>
                      {typeof value === 'number' ? (value * 100).toFixed(1) + '%' : value}
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ textAlign: 'right' }}>
                <span style={{
                  display: 'inline-block', padding: '4px 12px', borderRadius: '6px',
                  background: '#dcfce7', color: '#166534', fontWeight: '600', fontSize: '0.9rem'
                }}>
                  {exp.status}
                </span>
                <div style={{ marginTop: '4px', fontSize: '0.8rem', color: '#94a3b8' }}>{exp.timestamp}</div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ResultsPage

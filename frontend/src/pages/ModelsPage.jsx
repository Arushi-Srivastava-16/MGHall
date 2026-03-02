import React, { useState, useEffect } from 'react'
import axios from 'axios'
import './SharedStyles.css'

function ModelsPage() {
  const [models, setModels] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    axios.get('http://localhost:8000/api/models')
      .then(res => {
        setModels(res.data)
        setLoading(false)
      })
      .catch(err => {
        console.error("Models fetch failed", err)
        setLoading(false)
      })
  }, [])

  const MetricBar = ({ label, value, color }) => (
    <div style={{ marginBottom: '0.75rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
        <span style={{ fontSize: '0.75rem', color: '#64748b', fontWeight: '600' }}>{label}</span>
        <span style={{ fontSize: '0.75rem', color: '#1e293b', fontWeight: '700' }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div style={{ background: '#e2e8f0', borderRadius: '4px', height: '8px', overflow: 'hidden' }}>
        <div style={{ background: color, width: `${value * 100}%`, height: '100%', borderRadius: '4px', transition: 'width 0.5s ease' }}></div>
      </div>
    </div>
  )

  return (
    <div className="page">
      <h1>Active Models Registry</h1>
      <div className="card">
        <p>Currently loaded Neurosymbolic and LLM models available for inference.</p>
      </div>

      {loading ? (
        <p>Loading models...</p>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: '20px' }}>
          {models.map((model) => (
            <div key={model.model_id} className="card" style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                <span style={{ fontSize: '0.8rem', fontWeight: 'bold', color: '#64748b' }}>{model.provider}</span>
                <span style={{
                  background: model.status === 'Active' ? '#dcfce7' : '#f1f5f9',
                  color: model.status === 'Active' ? '#15803d' : '#64748b',
                  fontSize: '0.75rem', padding: '2px 8px', borderRadius: '99px', fontWeight: '600'
                }}>{model.status}</span>
              </div>

              <h3 style={{ marginBottom: '8px', fontSize: '1.2rem' }}>{model.model_name}</h3>
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.8rem', color: '#94a3b8', marginBottom: '16px' }}>{model.model_id}</div>

              <p style={{ flex: 1, fontSize: '0.95rem' }}>{model.description}</p>

              <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '12px', marginTop: '12px' }}>
                <h4 style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: '#64748b', margin: '0 0 8px 0' }}>Performance Metrics</h4>
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '8px' }}>
                  {Object.entries(model.metrics).map(([key, value]) => (
                    <div key={key}>
                      <span style={{ fontSize: '0.75rem', color: '#94a3b8', display: 'block' }}>{key.replace('_', ' ')}</span>
                      <span style={{ fontSize: '0.9rem', fontWeight: '600', color: '#334155' }}>{value}</span>
                    </div>
                  ))}
                </div>
              </div>

              {model.validation && (
                <div style={{ borderTop: '1px solid #e2e8f0', paddingTop: '12px', marginTop: '12px' }}>
                  <h4 style={{ fontSize: '0.8rem', textTransform: 'uppercase', color: '#64748b', margin: '0 0 12px 0' }}>
                    Validation Metrics <span style={{ fontSize: '0.7rem', color: '#94a3b8', fontWeight: 'normal' }}>({model.validation.test_samples} test samples)</span>
                  </h4>
                  <MetricBar label="Precision" value={model.validation.precision} color="#3b82f6" />
                  <MetricBar label="Recall" value={model.validation.recall} color="#8b5cf6" />
                  <MetricBar label="F1 Score" value={model.validation.f1_score} color="#10b981" />
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default ModelsPage

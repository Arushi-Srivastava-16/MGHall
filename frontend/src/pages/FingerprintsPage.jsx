import React, { useState } from 'react'
import { fingerprintsAPI } from '../services/api'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import './SharedStyles.css'

function FingerprintsPage() {
  const [chainId, setChainId] = useState('')
  const [fingerprint, setFingerprint] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleLoad = async () => {
    if (!chainId) return
    try {
      setLoading(true)
      const response = await fingerprintsAPI.get(chainId)
      setFingerprint(response.data)
    } catch (error) {
      console.error('Error loading fingerprint:', error)
      alert('Error loading fingerprint. Make sure the chain ID is valid.')
    } finally {
      setLoading(false)
    }
  }

  const prepareChartData = () => {
    if (!fingerprint || !fingerprint.features) return []
    
    const features = fingerprint.features
    return Object.entries(features)
      .filter(([key, value]) => typeof value === 'number')
      .map(([key, value]) => ({ name: key, value: value }))
      .slice(0, 10) // Top 10 features
  }

  return (
    <div className="page">
      <h1>Fingerprint Analysis</h1>

      <div className="card">
        <h2>Load Fingerprint</h2>
        <p style={{ marginBottom: '1rem', color: '#666' }}>
          <strong>How to get a Chain ID:</strong> Go to the Chains page, click on any chain ID to copy it, then paste it here.
        </p>
        <div style={{ display: 'flex', gap: '1rem', marginBottom: '1rem' }}>
          <input
            className="input"
            type="text"
            placeholder="Enter Chain ID (copy from Chains page)"
            value={chainId}
            onChange={(e) => setChainId(e.target.value)}
            style={{ flex: 1 }}
          />
          <button className="button" onClick={handleLoad} disabled={loading || !chainId}>
            {loading ? 'Loading...' : 'Load'}
          </button>
        </div>
      </div>

      {fingerprint && (
        <>
          <div className="card">
            <h2>Model Prediction</h2>
            {fingerprint.predicted_model ? (
              <div>
                <p><strong>Predicted Model:</strong> {fingerprint.predicted_model}</p>
                <p><strong>Confidence:</strong> {(fingerprint.confidence * 100).toFixed(2)}%</p>
              </div>
            ) : (
              <p>No classifier available for this domain</p>
            )}
          </div>

          <div className="card">
            <h2>Feature Values</h2>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={prepareChartData()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" fill="#3498db" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card">
            <h2>All Features</h2>
            <div className="features-grid">
              {Object.entries(fingerprint.features).map(([key, value]) => (
                <div key={key} className="feature-item">
                  <strong>{key}:</strong> {typeof value === 'number' ? value.toFixed(4) : value}
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}

export default FingerprintsPage


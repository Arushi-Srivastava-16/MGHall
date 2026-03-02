import React, { useState } from 'react'
import { trainingAPI } from '../services/api'
import './SharedStyles.css'

function TrainingPage() {
  const [domain, setDomain] = useState('math')
  const [modelType, setModelType] = useState('gat')
  const [jobId, setJobId] = useState(null)
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleStartTraining = async () => {
    try {
      setLoading(true)
      const response = await trainingAPI.start(domain, modelType, {})
      setJobId(response.data.job_id)
      setStatus(response.data)
      alert('Training job queued! Note: This is a placeholder - actual training should be run from command line.')
    } catch (error) {
      console.error('Error starting training:', error)
      alert('Error starting training: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  const checkStatus = async () => {
    if (!jobId) return
    try {
      const response = await trainingAPI.getStatus(jobId)
      setStatus(response.data)
    } catch (error) {
      console.error('Error checking status:', error)
    }
  }

  React.useEffect(() => {
    if (jobId) {
      const interval = setInterval(checkStatus, 2000)
      return () => clearInterval(interval)
    }
  }, [jobId])

  return (
    <div className="page">
      <h1>Training Interface</h1>

      <div className="card">
        <h2>Start Training</h2>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '1rem' }}>
          <div>
            <label>Domain:</label>
            <select className="select" value={domain} onChange={(e) => setDomain(e.target.value)}>
              <option value="math">Math</option>
              <option value="code">Code</option>
              <option value="medical">Medical</option>
            </select>
          </div>
          <div>
            <label>Model Type:</label>
            <select className="select" value={modelType} onChange={(e) => setModelType(e.target.value)}>
              <option value="gat">GAT</option>
              <option value="gcn">GCN</option>
            </select>
          </div>
        </div>
        <button className="button" onClick={handleStartTraining} disabled={loading}>
          {loading ? 'Starting...' : 'Start Training'}
        </button>
      </div>

      {status && (
        <div className="card">
          <h2>Training Status</h2>
          <p><strong>Job ID:</strong> {status.job_id}</p>
          <p><strong>Status:</strong> {status.status}</p>
          <p><strong>Progress:</strong> {(status.progress * 100).toFixed(1)}%</p>
          {status.current_epoch && (
            <p><strong>Epoch:</strong> {status.current_epoch} / {status.total_epochs}</p>
          )}
        </div>
      )}
    </div>
  )
}

export default TrainingPage


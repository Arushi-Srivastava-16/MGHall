import React, { useState } from 'react'
import { consensusAPI } from '../services/api'
import './SharedStyles.css'

function ConsensusPage() {
  const [strategy, setStrategy] = useState('majority_vote')
  const [strategies, setStrategies] = useState([])
  const [modelPredictions, setModelPredictions] = useState({
    'gpt-4': [true, true, false, true, true],
    'gemini-pro': [true, true, false, false, true],
    'llama-3-8b': [true, true, true, true, true],
  })
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  React.useEffect(() => {
    loadStrategies()
  }, [])

  const loadStrategies = async () => {
    try {
      const response = await consensusAPI.listStrategies()
      setStrategies(response.data.strategies || response.data || [])
      // If no strategies loaded, use defaults
      if (strategies.length === 0 && response.data.strategies) {
        setStrategies(response.data.strategies)
      }
    } catch (error) {
      console.error('Error loading strategies:', error)
      // Use default strategies if API fails
      setStrategies([
        { id: 'majority_vote', name: 'Majority Vote' },
        { id: 'weighted_vote', name: 'Weighted Vote' },
        { id: 'unanimous', name: 'Unanimous' },
        { id: 'expert_selection', name: 'Expert Selection' },
      ])
    }
  }

  const handleDetect = async () => {
    try {
      setLoading(true)
      const response = await consensusAPI.detect(modelPredictions, strategy)
      setResult(response.data)
    } catch (error) {
      console.error('Error detecting consensus:', error)
      alert('Error detecting consensus')
    } finally {
      setLoading(false)
    }
  }

  const updatePrediction = (model, stepIndex, value) => {
    setModelPredictions(prev => ({
      ...prev,
      [model]: prev[model].map((pred, idx) => idx === stepIndex ? value : pred)
    }))
  }

  return (
    <div className="page">
      <h1>Consensus Detection</h1>

      <div className="card">
        <h2>Configuration</h2>
        <div style={{ marginBottom: '1rem' }}>
          <label>Strategy:</label>
          <select className="select" value={strategy} onChange={(e) => setStrategy(e.target.value)}>
            {strategies.length > 0 ? (
              strategies.map(s => (
                <option key={s.id || s} value={s.id || s}>{s.name || s}</option>
              ))
            ) : (
              <>
                <option value="majority_vote">Majority Vote</option>
                <option value="weighted_vote">Weighted Vote</option>
                <option value="unanimous">Unanimous</option>
                <option value="expert_selection">Expert Selection</option>
              </>
            )}
          </select>
        </div>
        <button className="button" onClick={handleDetect} disabled={loading}>
          {loading ? 'Detecting...' : 'Detect Consensus'}
        </button>
      </div>

      <div className="card">
        <h2>Model Predictions</h2>
        <div className="predictions-table">
          <table>
            <thead>
              <tr>
                <th>Step</th>
                {Object.keys(modelPredictions).map(model => (
                  <th key={model}>{model}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Array.from({ length: 5 }).map((_, stepIdx) => (
                <tr key={stepIdx}>
                  <td>Step {stepIdx}</td>
                  {Object.entries(modelPredictions).map(([model, predictions]) => (
                    <td key={model}>
                      <input
                        type="checkbox"
                        checked={predictions[stepIdx] || false}
                        onChange={(e) => updatePrediction(model, stepIdx, e.target.checked)}
                      />
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {result && (
        <div className="card">
          <h2>Consensus Result</h2>
          <p><strong>Consensus Exists:</strong> {result.consensus_exists ? 'Yes' : 'No'}</p>
          <p><strong>Confidence:</strong> {(result.consensus_confidence * 100).toFixed(2)}%</p>
          <p><strong>Disagreement Points:</strong> {result.disagreement_points.join(', ') || 'None'}</p>
          <div>
            <strong>Step Consensus:</strong>
            <ul>
              {result.step_consensus.map((hasConsensus, idx) => (
                <li key={idx}>Step {idx}: {hasConsensus ? '✓' : '✗'}</li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}

export default ConsensusPage


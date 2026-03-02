import React, { useState, useEffect } from 'react'
import { inferenceAPI, modelsAPI } from '../services/api'
import './SharedStyles.css'

function InferencePage() {
  const [query, setQuery] = useState('Solve: 2x + 5 = 13')
  const [domain, setDomain] = useState('math')
  const [availableModels, setAvailableModels] = useState([])
  const [selectedModels, setSelectedModels] = useState([])
  const [temperature, setTemperature] = useState(0.7)
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    loadModels()
  }, [])

  const loadModels = async () => {
    try {
      const response = await modelsAPI.list()
      setAvailableModels(response.data)
      // Auto-select first 2 models
      if (response.data.length >= 2) {
        setSelectedModels([response.data[0].model_id, response.data[1].model_id])
      }
    } catch (error) {
      console.error('Error loading models:', error)
    }
  }

  const handleInference = async () => {
    if (!query || selectedModels.length === 0) {
      alert('Please enter a query and select at least one model')
      return
    }

    try {
      setLoading(true)
      const response = await inferenceAPI.generate(query, domain, selectedModels, temperature)
      setResults([response.data])
    } catch (error) {
      console.error('Error running inference:', error)
      alert('Error running inference: ' + (error.response?.data?.detail || error.message))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <h1>Real-Time Inference</h1>

      <div className="card">
        <h2>Configuration</h2>
        <div style={{ marginBottom: '1rem' }}>
          <label>Query:</label>
          <textarea
            className="input"
            rows="3"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Enter your query..."
          />
        </div>
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
            <label>Temperature:</label>
            <input
              className="input"
              type="number"
              min="0"
              max="2"
              step="0.1"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
            />
          </div>
        </div>
        <div style={{ marginBottom: '1rem' }}>
          <label>Models:</label>
          {availableModels.length === 0 ? (
            <div style={{ padding: '1rem', background: '#fff3cd', borderRadius: '4px', marginTop: '0.5rem' }}>
              <p><strong>No models available.</strong></p>
              <p>Make sure API keys are set in your .env file (GOOGLE_API_KEY for Gemini).</p>
            </div>
          ) : (
            <div className="models-checkboxes" style={{ marginTop: '0.5rem' }}>
              {availableModels.map((model) => (
                <label key={model.model_id} style={{ display: 'block', marginBottom: '0.5rem' }}>
                  <input
                    type="checkbox"
                    checked={selectedModels.includes(model.model_id)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedModels([...selectedModels, model.model_id])
                      } else {
                        setSelectedModels(selectedModels.filter(id => id !== model.model_id))
                      }
                    }}
                  />
                  {model.model_name} ({model.provider})
                </label>
              ))}
            </div>
          )}
        </div>
        <button className="button" onClick={handleInference} disabled={loading}>
          {loading ? 'Running Inference...' : 'Run Inference'}
        </button>
      </div>

      {results.length > 0 && (
        <div className="card">
          <h2>Results</h2>
          {results.map((result, idx) => (
            <div key={idx} style={{ marginBottom: '1rem', padding: '1rem', background: '#f9f9f9', borderRadius: '4px' }}>
              <p><strong>Model:</strong> {result.model_type}</p>
              <p><strong>Latency:</strong> {result.latency.toFixed(2)}s</p>
              {result.error ? (
                <p style={{ color: 'red' }}>Error: {result.error}</p>
              ) : (
                <div>
                  <strong>Steps:</strong>
                  <ol>
                    {result.steps.map((step, stepIdx) => (
                      <li key={stepIdx}>{step}</li>
                    ))}
                  </ol>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default InferencePage


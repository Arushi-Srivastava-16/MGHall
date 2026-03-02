import React, { useState, useEffect } from 'react'
import { chainsAPI } from '../services/api'
import ChainVisualizer from '../components/ChainVisualizer/ChainVisualizer'
import './ChainsPage.css'
import './SharedStyles.css'

function ChainsPage() {
  const [chains, setChains] = useState([])
  const [selectedChain, setSelectedChain] = useState(null)
  const [graphData, setGraphData] = useState(null)
  const [loading, setLoading] = useState(true)
  const [domain, setDomain] = useState('')

  useEffect(() => {
    loadChains()
  }, [domain])

  const loadChains = async () => {
    try {
      setLoading(true)
      const response = await chainsAPI.list(domain || null)
      setChains(response.data)
    } catch (error) {
      console.error('Error loading chains:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleChainSelect = async (chainId) => {
    try {
      const [chainResponse, graphResponse] = await Promise.all([
        chainsAPI.get(chainId),
        chainsAPI.getGraph(chainId),
      ])
      setSelectedChain(chainResponse.data)
      setGraphData(graphResponse.data)
    } catch (error) {
      console.error('Error loading chain:', error)
    }
  }

  return (
    <div className="page">
      <h1>Reasoning Chains</h1>

      <div className="page-controls">
        <select
          className="select"
          value={domain}
          onChange={(e) => setDomain(e.target.value)}
        >
          <option value="">All Domains</option>
          <option value="math">Math</option>
          <option value="code">Code</option>
          <option value="medical">Medical</option>
        </select>
        <button className="button" onClick={loadChains}>Refresh</button>
      </div>

      <div className="chains-layout">
        <div className="chains-list">
          <h2>Chains ({chains.length})</h2>
          {loading ? (
            <p>Loading chains...</p>
          ) : chains.length === 0 ? (
            <div className="card" style={{ marginTop: '1rem' }}>
              <p><strong>No chains found.</strong></p>
              <p>Make sure you have data files in:</p>
              <ul>
                <li><code>data/processed/splits/*.jsonl</code></li>
                <li><code>data/processed/code_test_splits/*.jsonl</code></li>
                <li><code>data/processed/medical_test_splits/*.jsonl</code></li>
              </ul>
            </div>
          ) : (
            <div className="chains-table">
              <table>
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Domain</th>
                    <th>Steps</th>
                    <th>Errors</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {chains.length === 0 ? (
                    <tr>
                      <td colSpan="5" style={{ textAlign: 'center', padding: '2rem' }}>
                        {loading ? 'Loading...' : 'No chains found. Make sure data files exist in data/processed/splits/'}
                      </td>
                    </tr>
                  ) : (
                    chains.map((chain) => (
                      <tr key={chain.chain_id}>
                        <td>
                          <span 
                            title={chain.chain_id}
                            style={{ cursor: 'pointer', textDecoration: 'underline' }}
                            onClick={() => {
                              navigator.clipboard.writeText(chain.chain_id)
                              alert('Chain ID copied to clipboard!')
                            }}
                          >
                            {chain.chain_id.substring(0, 20)}...
                          </span>
                        </td>
                        <td>{chain.domain}</td>
                        <td>{chain.num_steps}</td>
                        <td>{chain.error_count}</td>
                        <td>
                          <button
                            className="button"
                            onClick={() => handleChainSelect(chain.chain_id)}
                          >
                            View
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          )}
        </div>

        <div className="chain-detail">
          {selectedChain && (
            <div className="card">
              <h2>Chain Details</h2>
              <p><strong>Query:</strong> {selectedChain.query}</p>
              <p><strong>Domain:</strong> {selectedChain.domain}</p>
              <p><strong>Steps:</strong> {selectedChain.steps.length}</p>
              {graphData && <ChainVisualizer graphData={graphData} />}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default ChainsPage


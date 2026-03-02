/**
 * API client for backend communication.
 */

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Chains API
export const chainsAPI = {
  list: (domain, limit = 100) => 
    api.get('/api/chains', { params: { domain, limit } }),
  get: (chainId) => 
    api.get(`/api/chains/${chainId}`),
  getGraph: (chainId) => 
    api.get(`/api/chains/${chainId}/graph`),
}

// Models API
export const modelsAPI = {
  list: () => 
    api.get('/api/models'),
  getInfo: (modelId) => 
    api.get(`/api/models/${modelId}/info`),
  compare: (modelIds, domain) => 
    api.post('/api/models/compare', { model_ids: modelIds, domain }),
}

// Fingerprints API
export const fingerprintsAPI = {
  get: (chainId) => 
    api.get(`/api/fingerprints/${chainId}`),
  classify: (chainId) => 
    api.post('/api/fingerprints/classify', null, { params: { chain_id: chainId } }),
  getFeatureImportance: (domain = 'math') => 
    api.get('/api/fingerprints/features/importance', { params: { domain } }),
}

// Consensus API
export const consensusAPI = {
  listStrategies: () => 
    api.get('/api/consensus/strategies'),
  detect: (modelPredictions, strategy = 'majority_vote') => 
    api.post('/api/consensus/detect', { model_predictions: modelPredictions, strategy }),
}

// Patterns API
export const patternsAPI = {
  list: (domain, modelType, patternType) => 
    api.get('/api/patterns', { params: { domain, model_type: modelType, pattern_type: patternType } }),
  getStats: (domain) => 
    api.get('/api/patterns/stats', { params: { domain } }),
  get: (patternId) => 
    api.get(`/api/patterns/${patternId}`),
}

// Inference API
export const inferenceAPI = {
  generate: (query, domain, modelTypes, temperature = 0.7) => 
    api.post('/api/inference/generate', {
      query,
      domain,
      model_types: modelTypes,
      temperature,
    }),
  batch: (request) => 
    api.post('/api/inference/batch', request),
}

// Training API
export const trainingAPI = {
  start: (domain, modelType, config) => 
    api.post('/api/training/start', { domain, model_type: modelType, config }),
  getStatus: (jobId) => 
    api.get(`/api/training/status/${jobId}`),
  getResults: (jobId) => 
    api.get(`/api/training/results/${jobId}`),
}

// Results API
export const resultsAPI = {
  listExperiments: () => 
    api.get('/api/results/experiments'),
  getExperiment: (experimentId) => 
    api.get(`/api/results/${experimentId}`),
  getAggregateMetrics: () => 
    api.get('/api/results/metrics/aggregate'),
}

export default api


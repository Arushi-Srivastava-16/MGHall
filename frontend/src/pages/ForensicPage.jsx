import React, { useState } from 'react';
import { Search, FileText, Activity, AlertTriangle, CheckCircle, BarChart2, Microscope } from 'lucide-react';
import './SharedStyles.css';

function ForensicPage() {
    const [text, setText] = useState('');
    const [result, setResult] = useState(null);
    const [loading, setLoading] = useState(false);

    const analyzeText = async () => {
        if (!text) return;
        setLoading(true);
        try {
            const res = await fetch('http://localhost:8000/api/forensic', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text })
            });
            const data = await res.json();
            setResult(data);
        } catch (e) {
            console.error(e);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="page-container" style={{ padding: '40px', maxWidth: '800px', margin: '0 auto' }}>
            <div className="header-section" style={{ textAlign: 'center', marginBottom: '40px' }}>
                <h1 style={{ fontSize: '2.5rem', marginBottom: '10px', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '12px', color: '#0f172a' }}>
                    <Search size={40} className="text-blue-600" /> LLM Forensic Analysis
                </h1>
                <p style={{ fontSize: '1.2rem', color: '#64748b' }}>
                    Identify the source model of any generated text using <strong>Causal Fingerprinting</strong>.
                </p>
            </div>

            <div className="card" style={{ padding: '24px', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)' }}>
                <h3 style={{ marginBottom: '16px', color: '#1e293b', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <FileText size={20} /> Paste LLM-generated text:
                </h3>
                <textarea
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                    placeholder="Paste text here to analyze source signature..."
                    style={{
                        width: '100%', height: '150px', padding: '12px', borderRadius: '8px',
                        border: '1px solid #cbd5e1', fontFamily: 'monospace', fontSize: '1rem',
                        marginBottom: '20px', resize: 'vertical'
                    }}
                />
                <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                    <button
                        onClick={analyzeText}
                        disabled={loading || !text}
                        style={{
                            background: '#2563eb', color: 'white', padding: '12px 24px', borderRadius: '8px',
                            border: 'none', fontSize: '1rem', fontWeight: 'bold', cursor: 'pointer',
                            opacity: (loading || !text) ? 0.6 : 1, transition: 'background 0.2s',
                            display: 'flex', alignItems: 'center', gap: '8px'
                        }}
                    >
                        {loading ? <Activity className="spin" size={20} /> : <Microscope size={20} />}
                        {loading ? 'Analyzing Signature...' : 'Analyze Source'}
                    </button>
                </div>
            </div>

            {result && (
                <div className="card" style={{ marginTop: '30px', padding: '32px', border: '1px solid #e2e8f0', animation: 'fadeIn 0.5s ease-out' }}>

                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px' }}>
                        <h2 style={{ margin: 0, color: '#334155', display: 'flex', alignItems: 'center', gap: '10px' }}>
                            <Activity size={24} /> Forensic Results
                        </h2>
                        <div style={{ background: '#dbeafe', color: '#1e40af', padding: '8px 16px', borderRadius: '20px', fontWeight: 'bold' }}>
                            Most Likely: {result.verdict}
                        </div>
                    </div>

                    <div style={{ marginBottom: '40px' }}>
                        <h4 style={{ color: '#64748b', marginBottom: '16px', textTransform: 'uppercase', fontSize: '0.85rem', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <BarChart2 size={16} /> Model Probability Distribution
                        </h4>
                        {result.probabilities.map((prob, idx) => (
                            <div key={idx} style={{ marginBottom: '16px' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px', fontSize: '0.95rem', fontWeight: '600' }}>
                                    <span>{prob.model}</span>
                                    <span>{Math.round(prob.score * 100)}%</span>
                                </div>
                                <div style={{ width: '100%', height: '12px', background: '#f1f5f9', borderRadius: '6px', overflow: 'hidden' }}>
                                    <div style={{
                                        width: `${prob.score * 100}%`, height: '100%', background: prob.color,
                                        borderRadius: '6px', transition: 'width 1s ease-out'
                                    }}></div>
                                </div>
                            </div>
                        ))}
                    </div>

                    <div style={{ background: '#f8fafc', padding: '20px', borderRadius: '8px', border: '1px solid #e2e8f0' }}>
                        <h4 style={{ color: '#64748b', marginTop: 0, marginBottom: '16px', textTransform: 'uppercase', fontSize: '0.85rem', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '6px' }}>
                            <Search size={16} /> Signature Analysis
                        </h4>
                        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: '20px' }}>
                            <div>
                                <div style={{ fontSize: '0.85rem', color: '#64748b' }}>Error Propagation</div>
                                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#0f172a' }}>{result.signature.propagation} (High)</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.85rem', color: '#64748b' }}>Reasoning Depth</div>
                                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#0f172a' }}>{result.signature.depth} steps</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.85rem', color: '#64748b' }}>Error Pattern</div>
                                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#0f172a' }}>{result.signature.pattern}</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.85rem', color: '#64748b' }}>Certainty Score</div>
                                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#0f172a' }}>{result.signature.certainty}</div>
                            </div>
                            <div>
                                <div style={{ fontSize: '0.85rem', color: '#64748b' }}>Vocab Richness</div>
                                <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#0f172a' }}>{result.signature.richness}</div>
                            </div>
                        </div>
                    </div>

                </div>
            )}
        </div>
    );
}

export default ForensicPage;

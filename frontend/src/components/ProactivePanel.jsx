
import React from 'react'

function ProactivePanel({ riskScore }) {
    // Determine color and label based on score
    let color = '#3b82f6' // Default blue
    let riskLevel = 'Low'
    let labelColor = '#dcfce7'
    let textColor = '#166534'

    if (riskScore < 30) {
        color = '#22c55e' // Green
        riskLevel = 'Low Risk'
        labelColor = '#dcfce7'
        textColor = '#166534'
    } else if (riskScore < 70) {
        color = '#eab308' // Yellow
        riskLevel = 'Medium Risk'
        labelColor = '#fef9c3'
        textColor = '#854d0e'
    } else {
        color = '#ef4444' // Red
        riskLevel = 'High Risk'
        labelColor = '#fee2e2'
        textColor = '#991b1b'
    }

    // Calculate rotation for gauge needle (simulated)
    // 0 -> -90deg, 100 -> 90deg
    const rotation = (riskScore / 100) * 180 - 90

    return (
        <div className="card" style={{ padding: '16px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
                <h3 style={{ margin: 0, fontSize: '1rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
                    <span>⚠️</span> Proactive Prediction
                </h3>
                <span style={{
                    background: labelColor, color: textColor,
                    padding: '4px 8px', borderRadius: '6px', fontSize: '0.8rem', fontWeight: 'bold'
                }}>
                    {riskLevel}
                </span>
            </div>

            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                {/* Simple CSS Gauge */}
                <div style={{ position: 'relative', width: '120px', height: '60px', overflow: 'hidden', margin: '10px 0' }}>
                    <div style={{
                        width: '120px', height: '120px', borderRadius: '50%', background: '#e2e8f0',
                        position: 'absolute', top: 0, left: 0
                    }}></div>
                    <div style={{
                        width: '120px', height: '120px', borderRadius: '50%',
                        background: `conic-gradient(from 270deg, #22c55e 0%, #eab308 50%, #ef4444 100%)`,
                        position: 'absolute', top: 0, left: 0,
                        maskImage: 'linear-gradient(to bottom, transparent 50%, black 50%)',
                        WebkitMaskImage: 'linear-gradient(to bottom, transparent 50%, black 50%)',
                        transform: 'rotate(0deg)' // Fix rotation base
                    }}></div>

                    {/* Needle */}
                    <div style={{
                        width: '4px', height: '50px', background: '#475569', borderRadius: '2px',
                        position: 'absolute', bottom: '0', left: '58px', transformOrigin: 'bottom center',
                        transform: `rotate(${rotation}deg)`, transition: 'transform 1s cubic-bezier(0.4, 0, 0.2, 1)'
                    }}></div>
                </div>

                <div style={{ textAlign: 'center', marginTop: '8px' }}>
                    <div style={{ fontSize: '2rem', fontWeight: 'bold', color: color }}>{riskScore}%</div>
                    <p style={{ margin: 0, fontSize: '0.8rem', color: '#64748b' }}>Next Step Vulnerability</p>
                </div>

                {riskScore > 70 && (
                    <div style={{ marginTop: '16px', padding: '10px', background: '#fff1f2', border: '1px solid #fecdd3', borderRadius: '6px', fontSize: '0.85rem' }}>
                        <strong>Recommendation:</strong><br />
                        High complexity detected. Suggest enabling RAG or decomposition.
                    </div>
                )}
            </div>
        </div>
    )
}

export default ProactivePanel

import React from 'react';
import '../pages/SharedStyles.css';

const ConversationTimeline = ({ turns, onTurnClick }) => {
    if (!turns || turns.length === 0) return null;

    return (
        <div className="timeline-container" style={{ margin: '0 0 16px 0', padding: '12px', background: 'white', borderRadius: '12px', boxShadow: '0 1px 3px rgba(0,0,0,0.1)' }}>
            <h4 style={{ margin: '0 0 10px 0', fontSize: '0.9rem', color: '#64748b', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Conversation Timeline</h4>
            <div className="timeline-track" style={{ display: 'flex', gap: '8px', overflowX: 'auto', alignItems: 'center' }}>
                {turns.map((turn, idx) => (
                    <div key={turn.id} style={{ display: 'flex', alignItems: 'center' }}>
                        <div
                            onClick={() => onTurnClick && onTurnClick(idx)}
                            className={`timeline-node ${turn.hasError ? 'risk' : 'safe'} ${turn.isBlocked ? 'blocked' : ''}`}
                            style={{
                                width: '32px', height: '32px', borderRadius: '50%',
                                display: 'flex', alignItems: 'center', justifyContent: 'center',
                                fontSize: '0.8rem', fontWeight: 'bold', color: 'white',
                                background: turn.isBlocked ? '#94a3b8' : (turn.hasError ? '#ef4444' : (turn.isCorrection ? '#22c55e' : '#10b981')),
                                flexShrink: 0, position: 'relative', cursor: 'pointer', transition: 'transform 0.2s',
                                border: turn.isCorrection ? '2px solid #14532d' : 'none'
                            }}
                            title={`Turn ${turn.id}: ${turn.hasError ? 'Risk Detected' : 'Safe'}`}
                            onMouseOver={(e) => e.currentTarget.style.transform = 'scale(1.1)'}
                            onMouseOut={(e) => e.currentTarget.style.transform = 'scale(1)'}
                        >
                            {turn.id}
                        </div>
                        {idx < turns.length - 1 && (
                            <div style={{ width: '20px', height: '2px', background: '#cbd5e1', margin: '0 4px' }}></div>
                        )}
                    </div>
                ))}
            </div>
        </div>
    );
};

export default ConversationTimeline;

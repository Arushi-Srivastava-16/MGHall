import React, { useEffect, useRef } from 'react'
import { Network } from 'vis-network'
import './ChainVisualizer.css'

function ChainVisualizer({ graphData, onNodeClick }) {
  const networkRef = useRef(null)
  const containerRef = useRef(null)

  useEffect(() => {
    if (!graphData || !containerRef.current) return

    const data = {
      nodes: graphData.nodes || [],
      edges: graphData.edges || [],
    }

    const options = {
      nodes: {
        shape: 'box',
        font: { size: 14 },
        margin: 10,
      },
      edges: {
        arrows: { to: { enabled: true } },
        smooth: { type: 'cubicBezier' },
      },
      layout: {
        hierarchical: {
          direction: 'UD',
          sortMethod: 'directed',
        },
      },
      physics: {
        enabled: false,
      },
    }

    const network = new Network(containerRef.current, data, options)
    networkRef.current = network

    // Add Click Listener
    network.on("click", function (params) {
      if (params.nodes.length > 0 && onNodeClick) {
        onNodeClick(params.nodes[0]); // Pass the clicked Node ID
      }
    });

    return () => {
      if (networkRef.current) {
        networkRef.current.destroy()
      }
    }
  }, [graphData, onNodeClick])

  if (!graphData) {
    return <div className="chain-visualizer-loading">Loading graph...</div>
  }

  return (
    <div className="chain-visualizer">
      <div ref={containerRef} className="chain-visualizer-container" />
    </div>
  )
}

export default ChainVisualizer


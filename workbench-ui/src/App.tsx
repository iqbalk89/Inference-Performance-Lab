import { useEffect, useMemo, useState } from 'react'
import {
  Background,
  Controls,
  MarkerType,
  ReactFlow,
  type Edge,
  type NodeMouseHandler,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { ComponentNode, type ComponentFlowNode } from './ComponentNode'
import { Inspector } from './Inspector'
import type { ComponentData, ConnectionData, ScenarioData } from './types'

const nodeTypes = { component: ComponentNode }

export default function App() {
  const [scenario, setScenario] = useState<ScenarioData | null>(null)
  const [graphHistory, setGraphHistory] = useState<string[]>([])
  const [selectedComponent, setSelectedComponent] = useState<ComponentData | null>(null)
  const [selectedConnection, setSelectedConnection] = useState<ConnectionData | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/scenario.json')
      .then((response) => {
        if (!response.ok) throw new Error(`Scenario request failed: ${response.status}`)
        return response.json() as Promise<ScenarioData>
      })
      .then((loaded) => {
        setScenario(loaded)
        const requestedGraph = new URLSearchParams(window.location.search).get('graph')
        setGraphHistory([requestedGraph && loaded.diagrams[requestedGraph] ? requestedGraph : loaded.initial_graph_id])
      })
      .catch((reason: Error) => setError(reason.message))
  }, [])

  const graphId = graphHistory.at(-1)
  const diagram = scenario && graphId ? scenario.diagrams[graphId] : null

  const nodes = useMemo<ComponentFlowNode[]>(() => diagram?.components.map((component) => ({
    id: component.component_id,
    type: 'component',
    position: component.position,
    data: component,
  })) ?? [], [diagram])

  const edges = useMemo<Edge[]>(() => diagram?.connections.map((connection) => ({
    id: connection.connection_id,
    source: connection.source_id,
    target: connection.target_id,
    label: connection.category === 'mapping' ? '' : connection.badge,
    type: connection.category === 'mapping' ? 'straight' : 'smoothstep',
    animated: connection.category !== 'mapping',
    data: connection,
    markerEnd: { type: MarkerType.ArrowClosed },
    className: `edge-${connection.category}`,
    labelBgPadding: [7, 4],
    labelBgBorderRadius: 5,
    labelBgStyle: { fill: '#071323', fillOpacity: 0.96 },
    labelStyle: { fill: '#c8d8ec', fontWeight: 650 },
  })) ?? [], [diagram])

  const clearSelection = () => {
    setSelectedComponent(null)
    setSelectedConnection(null)
  }

  const pushInto: NodeMouseHandler<ComponentFlowNode> = (_, node) => {
    const target = node.data.drilldown_graph_id
    if (target && scenario?.diagrams[target]) {
      setGraphHistory((history) => [...history, target])
      clearSelection()
    }
  }

  if (error) return <main className="state-screen"><h1>Unable to load scenario</h1><p>{error}</p></main>
  if (!scenario || !diagram) return <main className="state-screen"><div className="loader"/><p>Loading workbench…</p></main>

  return (
    <main className="app-shell">
      <header className="topbar">
        <div>
          <div className="product-mark"><span>IPL</span> Inference System Performance Workbench</div>
          <h1>{diagram.title}</h1>
          <p>{diagram.subtitle}</p>
        </div>
        <div className="slice-badge"><span>SLICE</span><strong>00+</strong><small>Executable first model</small></div>
      </header>

      <nav className="breadcrumbs" aria-label="Diagram breadcrumb">
        {graphHistory.map((id, index) => (
          <button key={`${id}-${index}`} onClick={() => { setGraphHistory((history) => history.slice(0, index + 1)); clearSelection() }}>
            {scenario.diagrams[id].title}<span>{index < graphHistory.length - 1 ? '›' : ''}</span>
          </button>
        ))}
      </nav>

      <section className="workspace">
        <div className="canvas-panel">
          {graphId === 'gpu-0-detail' && <div className="lane-labels"><span>INFERENCE PROCESS</span><span>GPU HARDWARE</span></div>}
          {(graphId === 'prefill-detail' || graphId === 'decode-detail') && <div className="lane-labels phase-lanes"><span>LOGICAL TENSOR FLOW</span><span>PHYSICAL GPU HARDWARE</span></div>}
          <ReactFlow<ComponentFlowNode, Edge>
            nodes={nodes}
            edges={edges}
            nodeTypes={nodeTypes}
            onNodeClick={(_, node) => { setSelectedComponent(node.data); setSelectedConnection(null) }}
            onNodeDoubleClick={pushInto}
            onEdgeClick={(_, edge) => { setSelectedConnection(edge.data as unknown as ConnectionData); setSelectedComponent(null) }}
            onPaneClick={clearSelection}
            fitView
            fitViewOptions={{ padding: 0.16 }}
            minZoom={0.45}
            maxZoom={1.7}
          >
            <Background color="#263a55" gap={24} size={1} />
            <Controls showInteractive={false} />
          </ReactFlow>
          <div className="interaction-hint">Click to inspect · Double-click blocks labeled “Push in”</div>
        </div>

        <Inspector component={selectedComponent} connection={selectedConnection} />
      </section>

      <footer className="bottom-panel">
        <div><span>Scenario</span><strong>{scenario.title}</strong></div>
        <div><span>Memory implementation</span><strong>{String(scenario.metadata.memory_variant)}</strong></div>
        <div><span>Evidence state</span><strong>Theoretical + assumed</strong></div>
        <div className="assumptions"><span>Current assumptions</span><strong>{diagram.assumptions.join(' · ') || 'None recorded'}</strong></div>
      </footer>
    </main>
  )
}

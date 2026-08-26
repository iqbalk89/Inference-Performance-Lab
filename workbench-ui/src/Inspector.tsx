import type { ComponentData, ConnectionData } from './types'

interface InspectorProps {
  component: ComponentData | null
  connection: ConnectionData | null
}

export function Inspector({ component, connection }: InspectorProps) {
  if (!component && !connection) {
    return (
      <aside className="inspector empty-inspector">
        <div className="inspector-eyebrow">Inspector</div>
        <h2>Select a block or path</h2>
        <p>Inspect its purpose, quantities, evidence quality, and future calculation boundary.</p>
      </aside>
    )
  }

  if (connection) {
    return (
      <aside className="inspector">
        <div className="inspector-eyebrow">{connection.category} path</div>
        <h2>{connection.label}</h2>
        <dl className="properties">
          <div><dt>Source</dt><dd>{connection.source_id}</dd></div>
          <div><dt>Target</dt><dd>{connection.target_id}</dd></div>
          <div><dt>Direction</dt><dd>{connection.direction}</dd></div>
        </dl>
        <p className="notice">Slice 1 adds calculated traffic and rate values to executable paths.</p>
      </aside>
    )
  }

  return (
    <aside className="inspector">
      <div className="inspector-eyebrow">{component!.kind} · {component!.lane} lane</div>
      <h2>{component!.label}</h2>
      <p>{component!.summary}</p>
      {component!.metrics.length > 0 ? (
        <div className="metric-list">
          {component!.metrics.map((metric) => (
            <div className="metric-card" key={metric.name}>
              <div><span>{metric.name}</span><em className={`evidence ${metric.evidence}`}>{metric.evidence}</em></div>
              <strong>{metric.value}{metric.unit ? ` ${metric.unit}` : ''}</strong>
              {metric.description && <p>{metric.description}</p>}
              {metric.derivation && <code>{metric.derivation}</code>}
            </div>
          ))}
        </div>
      ) : <p className="notice">No quantitative metrics are assigned at Slice 0.</p>}
      {component!.drilldown_graph_id && <p className="notice">Double-click this block to push into its next level.</p>}
    </aside>
  )
}

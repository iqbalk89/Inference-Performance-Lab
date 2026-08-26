import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import type { ComponentData } from './types'

export type ComponentFlowNode = Node<ComponentData, 'component'>

export function ComponentNode({ id, data, selected }: NodeProps<ComponentFlowNode>) {
  const visibleMetrics = data.metrics.slice(0, 3)
  const drillable = Boolean(data.drilldown_graph_id)

  return (
    <div data-testid={`component-${id}`} className={`component-node kind-${data.kind} lane-${data.lane} ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <div className="node-kind">{data.kind}</div>
      <div className="node-title">{data.label}</div>
      {visibleMetrics.length > 0 && <div className="node-metrics">
        {visibleMetrics.map((metric) => (
          <div className="node-metric" key={metric.name}>
            <span>{metric.name}</span>
            <strong>{metric.value}{metric.unit ? ` ${metric.unit}` : ''}</strong>
          </div>
        ))}
      </div>}
      {drillable && <div className="push-in">Push in <span>→</span></div>}
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

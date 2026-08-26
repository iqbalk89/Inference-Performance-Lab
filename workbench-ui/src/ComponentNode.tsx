import { Handle, Position, type Node, type NodeProps } from '@xyflow/react'
import type { ComponentData } from './types'

export type ComponentFlowNode = Node<ComponentData, 'component'>

export function ComponentNode({ id, data, selected }: NodeProps<ComponentFlowNode>) {
  const primaryMetric = data.metrics[0]
  const drillable = Boolean(data.drilldown_graph_id)

  return (
    <div data-testid={`component-${id}`} className={`component-node kind-${data.kind} lane-${data.lane} ${selected ? 'selected' : ''}`}>
      <Handle type="target" position={Position.Left} />
      <div className="node-kind">{data.kind}</div>
      <div className="node-title">{data.label}</div>
      {primaryMetric && (
        <div className="node-metric">
          <span>{primaryMetric.name}</span>
          <strong>{primaryMetric.value}{primaryMetric.unit ? ` ${primaryMetric.unit}` : ''}</strong>
        </div>
      )}
      {drillable && <div className="push-in">Push in <span>→</span></div>}
      <Handle type="source" position={Position.Right} />
    </div>
  )
}

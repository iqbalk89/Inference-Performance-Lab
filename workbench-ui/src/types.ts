export type EvidenceKind = 'theoretical' | 'assumed' | 'calibrated' | 'measured'

export interface CalculationInput {
  symbol: string
  value: string
  meaning: string
  source: string
}

export interface CalculationStep {
  title: string
  expression: string
  explanation: string
}

export interface CalculationDetail {
  title: string
  concept: string
  formula: string
  inputs: CalculationInput[]
  steps: CalculationStep[]
  unit_analysis: string
  interpretation: string
  assumptions: string[]
}

export interface Metric {
  name: string
  value: string | number
  unit: string
  evidence: EvidenceKind
  description: string
  derivation: string
  calculation: CalculationDetail | null
}

export interface ComponentData {
  [key: string]: unknown
  component_id: string
  label: string
  kind: string
  summary: string
  position: { x: number; y: number }
  metrics: Metric[]
  drilldown_graph_id: string | null
  lane: 'process' | 'hardware' | 'analysis'
}

export interface ConnectionData {
  [key: string]: unknown
  connection_id: string
  source_id: string
  target_id: string
  label: string
  direction: string
  category: string
  metrics: Metric[]
  badge: string
}

export interface DiagramData {
  graph_id: string
  title: string
  subtitle: string
  parent_graph_id: string | null
  components: ComponentData[]
  connections: ConnectionData[]
  assumptions: string[]
  charts: ChartData[]
}

export interface ChartPoint {
  x: number
  label: string
  arithmetic_intensity: number
  compute_time_us: number
  memory_time_us: number
  lower_bound_us: number
  bottleneck: string
}

export interface ChartData {
  chart_id: string
  kind: string
  title: string
  description: string
  x_label: string
  y_label: string
  points: ChartPoint[]
  parameters: Record<string, number>
}

export interface ScenarioData {
  scenario_id: string
  title: string
  initial_graph_id: string
  diagrams: Record<string, DiagramData>
  metadata: Record<string, unknown>
}

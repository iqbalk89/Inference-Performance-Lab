import { useMemo, useState } from 'react'
import type { ChartData, ChartPoint } from './types'

interface RooflineChartProps { chart: ChartData }

function pointFor(rows: number, computeTflops: number, bandwidthGbps: number, chart: ChartData): ChartPoint {
  const k = chart.parameters.input_width
  const n = chart.parameters.output_width
  const bytesPerValue = chart.parameters.bytes_per_value
  const flops = 2 * rows * k * n
  const bytes = k * n * bytesPerValue + rows * k * bytesPerValue + rows * n * bytesPerValue
  const arithmeticIntensity = flops / bytes
  const computeTime = flops / (computeTflops * 1e12) * 1e6
  const memoryTime = bytes / (bandwidthGbps * 1e9) * 1e6
  return { x: rows, label: String(rows), arithmetic_intensity: arithmeticIntensity, compute_time_us: computeTime, memory_time_us: memoryTime, lower_bound_us: Math.max(computeTime, memoryTime), bottleneck: memoryTime > computeTime ? 'HBM bandwidth' : 'FP16 compute' }
}

export function RooflineChart({ chart }: RooflineChartProps) {
  const [computeTflops, setComputeTflops] = useState(chart.parameters.compute_tflops)
  const [bandwidthGbps, setBandwidthGbps] = useState(chart.parameters.hbm_bandwidth_gbps)
  const [selectedRows, setSelectedRows] = useState(chart.parameters.selected_rows)
  const points = useMemo(() => chart.points.map((point) => pointFor(point.x, computeTflops, bandwidthGbps, chart)), [chart, computeTflops, bandwidthGbps])
  const selected = pointFor(selectedRows, computeTflops, bandwidthGbps, chart)
  const width = 720
  const height = 240
  const pad = { left: 58, right: 18, top: 20, bottom: 38 }
  const maxY = Math.max(...points.map((point) => Math.max(point.compute_time_us, point.memory_time_us)), selected.lower_bound_us) * 1.12
  const minLog = Math.log2(points[0].x)
  const maxLog = Math.log2(points[points.length - 1].x)
  const x = (value: number) => pad.left + (Math.log2(value) - minLog) / (maxLog - minLog) * (width - pad.left - pad.right)
  const y = (value: number) => height - pad.bottom - value / maxY * (height - pad.top - pad.bottom)
  const line = (field: 'compute_time_us' | 'memory_time_us') => points.map((point) => `${x(point.x)},${y(point[field])}`).join(' ')
  const selectedX = x(selectedRows)

  return <section className="chart-card">
    <div className="chart-heading"><div><div className="inspector-eyebrow">Interactive performance model</div><h2>{chart.title}</h2><p>{chart.description}</p></div></div>
    <div className="chart-controls">
      <label>Selected M<input type="number" min="1" max="16384" value={selectedRows} onChange={(event) => setSelectedRows(Math.max(1, Number(event.target.value) || 1))} /></label>
      <label>HBM bandwidth (GB/s)<input type="number" min="1" value={bandwidthGbps} onChange={(event) => setBandwidthGbps(Math.max(1, Number(event.target.value) || 1))} /></label>
      <label>Peak FP16 (TFLOP/s)<input type="number" min="0.1" value={computeTflops} onChange={(event) => setComputeTflops(Math.max(0.1, Number(event.target.value) || 0.1))} /></label>
    </div>
    <div className="chart-layout">
      <svg className="roofline-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label="Ideal compute and memory latency bounds as token rows increase">
        <line x1={pad.left} x2={width - pad.right} y1={height - pad.bottom} y2={height - pad.bottom} className="chart-axis" />
        <line x1={pad.left} x2={pad.left} y1={pad.top} y2={height - pad.bottom} className="chart-axis" />
        <polyline points={line('memory_time_us')} className="chart-memory-line" />
        <polyline points={line('compute_time_us')} className="chart-compute-line" />
        <line x1={selectedX} x2={selectedX} y1={pad.top} y2={height - pad.bottom} className="chart-selected-line" />
        {points.filter((point) => point.x === 1 || point.x === 16 || point.x === 128 || point.x === 512 || point.x === 2048).map((point) => <text key={point.x} x={x(point.x)} y={height - 12} textAnchor="middle" className="chart-label">{point.label}</text>)}
        <text x={12} y={height / 2} transform={`rotate(-90 12 ${height / 2})`} className="chart-axis-label">µs</text>
        <text x={width / 2} y={height - 1} textAnchor="middle" className="chart-axis-label">{chart.x_label}</text>
        <text x={selectedX + 6} y={pad.top + 12} className="chart-selected-label">M={selectedRows}</text>
      </svg>
      <div className="chart-legend"><span><i className="legend-memory" />Memory bound</span><span><i className="legend-compute" />Compute bound</span></div>
    </div>
    <div className="chart-results">
      <div><span>Arithmetic intensity</span><strong>{selected.arithmetic_intensity.toFixed(2)} FLOPs/byte</strong></div>
      <div><span>Compute bound</span><strong>{selected.compute_time_us.toFixed(3)} µs</strong></div>
      <div><span>Memory bound</span><strong>{selected.memory_time_us.toFixed(3)} µs</strong></div>
      <div><span>Roofline lower bound</span><strong>{selected.lower_bound_us.toFixed(3)} µs · {selected.bottleneck}</strong></div>
    </div>
    <p className="chart-footnote">The lines are ideal ceilings from the simplified Problem 02 model. They exclude kernel launch overhead, cache effects, scheduling, and achieved-utilization losses.</p>
  </section>
}

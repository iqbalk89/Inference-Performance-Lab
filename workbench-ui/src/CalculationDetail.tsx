import type { CalculationDetail as CalculationDetailData } from './types'

interface CalculationDetailProps {
  calculation: CalculationDetailData
  onClose: () => void
}

export function CalculationDetail({ calculation, onClose }: CalculationDetailProps) {
  return (
    <div className="calculation-overlay" role="dialog" aria-modal="true" aria-label={calculation.title}>
      <section className="calculation-sheet">
        <header className="calculation-header">
          <div>
            <div className="inspector-eyebrow">Complete calculation walkthrough</div>
            <h2>{calculation.title}</h2>
          </div>
          <button className="close-calculation" onClick={onClose} aria-label="Close calculation">×</button>
        </header>

        <div className="calculation-content">
          <section className="concept-section">
            <h3>What are we calculating?</h3>
            <p>{calculation.concept}</p>
          </section>

          <section>
            <h3>General formula</h3>
            <div className="formula-display">{calculation.formula}</div>
          </section>

          <section>
            <h3>Where every input comes from</h3>
            <div className="input-table">
              <div className="input-table-header"><span>Symbol</span><span>Value</span><span>Meaning</span><span>Source</span></div>
              {calculation.inputs.map((input, index) => (
                <div className="input-table-row" key={`${input.symbol}-${index}`}>
                  <code>{input.symbol}</code><strong>{input.value}</strong><span>{input.meaning}</span><span>{input.source}</span>
                </div>
              ))}
            </div>
          </section>

          <section>
            <h3>Calculation, one step at a time</h3>
            <ol className="calculation-steps">
              {calculation.steps.map((step, index) => (
                <li key={`${step.title}-${index}`}>
                  <div className="step-number">{index + 1}</div>
                  <div><h4>{step.title}</h4><div className="step-expression">{step.expression}</div><p>{step.explanation}</p></div>
                </li>
              ))}
            </ol>
          </section>

          <section className="unit-section">
            <h3>Why the units work</h3>
            <p>{calculation.unit_analysis}</p>
          </section>

          <section className="interpretation-section">
            <h3>What the result means</h3>
            <p>{calculation.interpretation}</p>
          </section>

          {calculation.assumptions.length > 0 && <section>
            <h3>Assumptions and limitations</h3>
            <ul className="calculation-assumptions">{calculation.assumptions.map((item) => <li key={item}>{item}</li>)}</ul>
          </section>}
        </div>
      </section>
    </div>
  )
}

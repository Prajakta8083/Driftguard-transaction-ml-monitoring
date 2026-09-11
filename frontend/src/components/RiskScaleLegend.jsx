export default function RiskScaleLegend() {
  return (
    <div className="risk-scale">
      <span className="risk-scale-item">
        <span className="dot" style={{ background: "var(--status-healthy)" }} />
        <strong>0–50%</strong>&nbsp;Legitimate
      </span>
      <span className="risk-scale-item">
        <span className="dot" style={{ background: "var(--status-warning)" }} />
        <strong>50–80%</strong>&nbsp;Hold for review
      </span>
      <span className="risk-scale-item">
        <span className="dot" style={{ background: "var(--status-critical)" }} />
        <strong>80–100%</strong>&nbsp;Block &amp; escalate
      </span>
    </div>
  );
}

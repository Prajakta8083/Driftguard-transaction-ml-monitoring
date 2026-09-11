export default function ShapBars({ drivers }) {
  if (!drivers || drivers.length === 0) {
    return <p className="empty-state">No SHAP drivers available.</p>;
  }
  const maxAbs = Math.max(...drivers.map((d) => Math.abs(d.shap_value)), 0.0001);

  return (
    <div>
      {drivers.map((d) => {
        const pct = (Math.abs(d.shap_value) / maxAbs) * 100;
        const isPositive = d.shap_value > 0;
        return (
          <div key={d.feature} style={{ marginBottom: 12 }}>
            <div className="shap-row" style={{ marginBottom: 3 }}>
              <span className="shap-feature-name">{d.feature}</span>
              <div className="shap-bar-track">
                <div
                  className={`shap-bar-fill ${isPositive ? "positive" : "negative"}`}
                  style={{ width: `${pct}%` }}
                />
              </div>
              <span className="shap-value-label">{d.shap_value.toFixed(4)}</span>
            </div>
            <div style={{ marginLeft: 70 }}>
              <span className={`driver-direction ${isPositive ? "up" : "down"}`}>
                {isPositive ? "↑ increases fraud risk" : "↓ decreases fraud risk"}
              </span>
            </div>
          </div>
        );
      })}
      <p style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 10 }}>
        <span style={{ color: "var(--status-critical)" }}>■</span> pushes toward fraud &nbsp;&nbsp;
        <span style={{ color: "var(--brand)" }}>■</span> pushes toward legitimate
      </p>
    </div>
  );
}

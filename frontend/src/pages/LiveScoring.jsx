import { useState } from "react";
import { Fingerprint } from "lucide-react";
import { api } from "../api";
import ShapBars from "../components/ShapBars";
import PageHeader from "../components/PageHeader";
import RiskScaleLegend from "../components/RiskScaleLegend";
import { buildPlainExplanation } from "../utils/explainText";

function recommendationFor(probability, predictedLabel) {
  if (predictedLabel === 1 && probability >= 0.8) {
    return { tone: "critical", label: "Recommended action", text: "Block transaction and escalate to fraud review immediately." };
  }
  if (predictedLabel === 1) {
    return { tone: "warning", label: "Recommended action", text: "Hold for manual review before approving." };
  }
  return { tone: "healthy", label: "Recommended action", text: "No action needed — approve normally." };
}

const FEATURE_FIELDS = ["Time", ...Array.from({ length: 28 }, (_, i) => `V${i + 1}`), "Amount"];

function randomSamplePayload() {
  const payload = { Time: Math.round(Math.random() * 100000), Amount: +(Math.random() * 300).toFixed(2) };
  for (let i = 1; i <= 28; i++) payload[`V${i}`] = +((Math.random() - 0.5) * 4).toFixed(3);
  return payload;
}

export default function LiveScoring() {
  const [form, setForm] = useState(randomSamplePayload());
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [feedbackSent, setFeedbackSent] = useState(false);

  function updateField(key, value) {
    setForm((f) => ({ ...f, [key]: value === "" ? "" : Number(value) }));
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setFeedbackSent(false);
    try {
      const res = await api.postTransaction(form);
      setResult(res);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function sendFeedback(actualLabel) {
    if (!result) return;
    await api.postFeedback(result.transaction_id, actualLabel);
    setFeedbackSent(true);
  }

  const isFraud = result?.predicted_label === 1;

  return (
    <div>
      <PageHeader
        eyebrow="FRAUD DETECTION"
        title="Live Scoring"
        subtitle="Enter a transaction's details below and the model will score it in real time, showing exactly which factors drove the decision."
      />

      {error && (
        <div className="error-banner">
          Couldn't reach the backend to score this transaction.
          <span className="hint">
            Make sure the FastAPI server is running at http://localhost:8000, then try again. ({error})
          </span>
        </div>
      )}

      <div className="panel-row">
        <div className="panel-col panel">
          <p className="panel-title">Transaction details</p>
          <p className="panel-subtext">
            This form simulates one transaction. In a real system, a payment processor would send
            these same numbers automatically the instant a card is swiped — nobody types them by hand.
            Here, you enter them manually so you can test the model with different values.
          </p>
          <form onSubmit={handleSubmit}>
            <div className="form-grid" style={{ gridTemplateColumns: "repeat(2, 1fr)", marginBottom: 4 }}>
              <div className="field">
                <label>Time (seconds since first transaction)</label>
                <input type="number" step="any" value={form.Time} onChange={(e) => updateField("Time", e.target.value)} />
              </div>
              <div className="field">
                <label>Amount</label>
                <input type="number" step="any" value={form.Amount} onChange={(e) => updateField("Amount", e.target.value)} />
              </div>
            </div>

            <details className="advanced-fields" open>
              <summary>Advanced: anonymized risk factors (V1–V28)</summary>
              <div className="plain-explanation" style={{ marginBottom: 14 }}>
                <strong>What are V1–V28?</strong> The original dataset owner ran the raw transaction
                data (card type, location, merchant, etc.) through a privacy transformation called PCA,
                which mixes everything together into 28 unlabeled numbers. That's why there's no
                individual definition for "V14" the way there is for "Amount" — the real meaning was
                deliberately destroyed to protect customer data, and only the model can make sense of
                the pattern across all 28 together. <strong>The order matters too</strong>: the model
                was trained expecting V1 first, then V2, then V3... in that exact sequence — the app
                enforces this automatically so you never have to think about it.
              </div>
              <div className="form-grid">
                {FEATURE_FIELDS.filter((k) => k !== "Time" && k !== "Amount").map((key) => (
                  <div className="field" key={key}>
                    <label>{key}</label>
                    <input
                      type="number"
                      step="any"
                      value={form[key]}
                      onChange={(e) => updateField(key, e.target.value)}
                    />
                  </div>
                ))}
              </div>
            </details>

            <div style={{ display: "flex", gap: 10, marginTop: 16 }}>
              <button type="submit" className="btn-primary" disabled={loading}>
                {loading ? "Scoring…" : "Score transaction"}
              </button>
              <button type="button" className="btn-secondary" onClick={() => setForm(randomSamplePayload())}>
                Fill random sample
              </button>
            </div>
          </form>
        </div>

        <div className="panel-col panel">
          <p className="panel-title">Result</p>
          {result && (
            <p className="panel-subtext" style={{ marginBottom: 16 }}>
              Scored by: <strong>{result.model_version}</strong> — the model currently active in this
              app (see the Model Report page for how it was chosen).
            </p>
          )}
          {!result && (
            <p className="empty-state">
              <Fingerprint size={22} style={{ marginBottom: 8, opacity: 0.4 }} />
              <br />
              Submit a transaction to see the model's decision here.
            </p>
          )}
          {result && (
            <div>
              <p className="kpi-label">Fraud probability</p>
              <p className={`hero-metric-value ${isFraud ? "critical" : "healthy"}`}>
                {(result.fraud_probability * 100).toFixed(1)}%
              </p>
              <span className={`risk-pill ${isFraud ? "critical" : "healthy"}`}>
                {isFraud ? "Fraud" : "Legitimate"}
              </span>
              <div className="progress-track">
                <div
                  className={`progress-fill ${isFraud ? "critical" : "healthy"}`}
                  style={{ width: `${Math.min(result.fraud_probability * 100, 100)}%` }}
                />
              </div>
              <div style={{ marginTop: 12, marginBottom: 4 }}>
                <RiskScaleLegend />
              </div>

              {(() => {
                const rec = recommendationFor(result.fraud_probability, result.predicted_label);
                return (
                  <div className={`recommendation-box ${rec.tone}`} style={{ marginTop: 20 }}>
                    <div>
                      <p className="recommendation-label">{rec.label}</p>
                      <p className="recommendation-text">{rec.text}</p>
                    </div>
                  </div>
                );
              })()}

              <div className="plain-explanation">
                {buildPlainExplanation(result.top_shap_drivers, result.predicted_label)}
              </div>

              <p className="panel-title">Top reasons behind this decision</p>
              <p className="panel-subtext">
                Each bar shows how much that factor pushed the score toward fraud (red) or toward
                legitimate (blue). Longer bars mattered more.
              </p>
              <ShapBars drivers={result.top_shap_drivers} />

              <p style={{ fontSize: 12, color: "var(--text-faint)", marginTop: 14 }}>
                Transaction #{result.transaction_id}
              </p>

              <div style={{ marginTop: 16, paddingTop: 16, borderTop: "1px solid var(--border)" }}>
                <p style={{ fontSize: 12.5, color: "var(--text-muted)", marginBottom: 8 }}>
                  In real fraud systems, the true outcome (fraud or not) often isn't known until days
                  later — e.g. after a chargeback. Simulate that here:
                </p>
                <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <button className="btn-secondary" onClick={() => sendFeedback(1)}>
                    Later confirmed: Fraud
                  </button>
                  <button className="btn-secondary" onClick={() => sendFeedback(0)}>
                    Later confirmed: Legitimate
                  </button>
                  {feedbackSent && <span style={{ fontSize: 12, color: "var(--status-healthy)", fontWeight: 600 }}>Recorded ✓</span>}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

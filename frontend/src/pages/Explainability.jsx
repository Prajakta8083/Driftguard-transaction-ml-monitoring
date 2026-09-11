import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";
import { api } from "../api";
import ShapBars from "../components/ShapBars";
import PageHeader from "../components/PageHeader";
import { buildPlainExplanation } from "../utils/explainText";

export default function Explainability() {
  const [globalData, setGlobalData] = useState(null);
  const [error, setError] = useState(null);

  const [txnId, setTxnId] = useState("");
  const [localResult, setLocalResult] = useState(null);
  const [localError, setLocalError] = useState(null);
  const [loadingLocal, setLoadingLocal] = useState(false);

  useEffect(() => {
    api.getGlobalImportance().then(setGlobalData).catch((err) => setError(err.message));
  }, []);

  async function lookupLocal(e) {
    e.preventDefault();
    if (!txnId) return;
    setLoadingLocal(true);
    setLocalError(null);
    try {
      const res = await api.explainPrediction(txnId);
      setLocalResult(res);
    } catch (err) {
      setLocalError(err.message);
      setLocalResult(null);
    } finally {
      setLoadingLocal(false);
    }
  }

  const chartData = globalData ? globalData.features.slice(0, 12).map((f) => ({ feature: f.feature, importance: f.importance })) : [];

  return (
    <div>
      <PageHeader
        eyebrow="MODEL TRANSPARENCY"
        title="Explainability"
        subtitle="Global view: what the model relies on across every prediction. Local view: why one specific transaction was scored the way it was."
      />

      <div className="panel" style={{ marginBottom: 20 }}>
        <p className="panel-title">What the model pays attention to overall</p>
        <p className="panel-subtext">
          Longer bars mean the model leans on that feature more heavily, on average, across many
          transactions — this doesn't say whether a feature pushes toward fraud or not, only how much
          it matters.
        </p>
        {error && (
          <div className="error-banner">
            Couldn't load global feature importance.
            <span className="hint">Make sure the FastAPI server is running at http://localhost:8000. ({error})</span>
          </div>
        )}
        {!globalData && !error && <p className="empty-state">Loading…</p>}
        {globalData && chartData.length === 0 && (
          <p className="empty-state">No global importance available yet (needs test.csv next to the backend).</p>
        )}
        {chartData.length > 0 && (
          <ResponsiveContainer width="100%" height={340}>
            <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" stroke="var(--text-muted)" fontSize={11} />
              <YAxis type="category" dataKey="feature" stroke="var(--text-muted)" fontSize={11} width={50} />
              <Tooltip contentStyle={{ background: "white", border: "1px solid var(--border)" }} />
              <Bar dataKey="importance" fill="#4c6fff" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>

      <div className="panel">
        <p className="panel-title">Why was one specific transaction flagged?</p>
        <p className="panel-subtext">
          Paste a Prediction ID from the History page to see the exact reasoning behind that single decision.
        </p>
        <form onSubmit={lookupLocal} style={{ display: "flex", gap: 10, marginBottom: 16 }}>
          <input
            type="number"
            placeholder="Transaction ID"
            value={txnId}
            onChange={(e) => setTxnId(e.target.value)}
            style={{
              background: "#fbfcfe",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              padding: "9px 12px",
              color: "var(--text-primary)",
              width: 180,
            }}
          />
          <button className="btn-primary" type="submit" disabled={loadingLocal}>
            {loadingLocal ? "Looking up…" : "Explain"}
          </button>
        </form>

        {localError && (
          <div className="error-banner">
            No prediction found for that ID.
            <span className="hint">{localError}</span>
          </div>
        )}
        {!localResult && !localError && (
          <p className="empty-state">Enter a transaction ID from Prediction History to see why it was scored that way.</p>
        )}
        {localResult && (
          <div>
            <div className="plain-explanation">
              {buildPlainExplanation(localResult.top_shap_drivers, localResult.predicted_label)}
            </div>
            <ShapBars drivers={localResult.top_shap_drivers} />
            <p style={{ fontSize: 12, color: "var(--text-faint)", marginTop: 10 }}>
              Fraud probability: {(localResult.fraud_probability * 100).toFixed(1)}% · Model: {localResult.model_version}
            </p>
          </div>
        )}
      </div>

      <p style={{ fontSize: 12, color: "var(--text-faint)", marginTop: 16, display: "flex", gap: 8, alignItems: "flex-start" }}>
        <span>ℹ️</span>
        Model contribution only — these values show which input factors influenced the score. They
        don't identify real-world causes like a stolen card or a compromised merchant; that judgment
        still requires human review.
      </p>
    </div>
  );
}

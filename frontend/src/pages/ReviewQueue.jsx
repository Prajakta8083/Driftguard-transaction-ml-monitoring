import { useEffect, useState } from "react";
import { ClipboardList } from "lucide-react";
import { api } from "../api";
import PageHeader from "../components/PageHeader";
import KpiCard from "../components/KpiCard";
import ShapBars from "../components/ShapBars";
import { buildPlainExplanation } from "../utils/explainText";

export default function ReviewQueue() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [resolvingId, setResolvingId] = useState(null);

  async function load() {
    try {
      const res = await api.getReviewQueue();
      setData(res);
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    load();
  }, []);

  async function resolve(transactionId, actualLabel) {
    setResolvingId(transactionId);
    try {
      await api.postFeedback(transactionId, actualLabel);
      await load();
    } finally {
      setResolvingId(null);
    }
  }

  return (
    <div>
      <PageHeader
        eyebrow="ANALYST WORKFLOW"
        title="Review Queue"
        subtitle="Every transaction the model flagged as fraud that hasn't been confirmed one way or the other yet. Resolving one here removes it from the queue and feeds Model Health."
      />

      {error && (
        <div className="error-banner">
          Couldn't load the review queue.
          <span className="hint">Make sure the FastAPI server is running at http://localhost:8000. ({error})</span>
        </div>
      )}

      <div className="kpi-row">
        <KpiCard icon={ClipboardList} label="Awaiting review" value={data?.total_pending ?? "—"} tone={data?.total_pending > 0 ? "critical" : "healthy"} />
      </div>

      <div className="panel">
        <p className="panel-title">Pending cases, highest risk first</p>
        {!data && !error && <p className="empty-state">Loading…</p>}
        {data && data.items.length === 0 && (
          <p className="empty-state">Nothing pending — every flagged transaction has been resolved.</p>
        )}
        {data &&
          data.items.map((item) => (
            <div className="queue-card" key={item.prediction_id}>
              <div className="queue-card-top">
                <div>
                  <p style={{ fontWeight: 700, fontSize: 15, margin: "0 0 2px 0" }}>
                    Transaction #{item.transaction_id}
                  </p>
                  <p style={{ fontSize: 12.5, color: "var(--text-faint)", margin: 0 }}>
                    Flagged {new Date(item.created_at).toLocaleString()}
                  </p>
                </div>
                <p className="mono" style={{ fontSize: 22, fontWeight: 700, color: "var(--status-critical)", margin: 0 }}>
                  {(item.fraud_probability * 100).toFixed(1)}%
                </p>
              </div>
              <div className="plain-explanation" style={{ marginBottom: 10 }}>
                {buildPlainExplanation(item.top_shap_drivers, 1)}
              </div>
              <ShapBars drivers={item.top_shap_drivers.slice(0, 3)} />
              <div className="queue-card-actions">
                <button
                  className="btn-primary"
                  disabled={resolvingId === item.transaction_id}
                  onClick={() => resolve(item.transaction_id, 1)}
                  style={{ background: "var(--status-critical)" }}
                >
                  Confirm fraud
                </button>
                <button
                  className="btn-secondary"
                  disabled={resolvingId === item.transaction_id}
                  onClick={() => resolve(item.transaction_id, 0)}
                >
                  Clear as legitimate
                </button>
              </div>
            </div>
          ))}
      </div>
    </div>
  );
}

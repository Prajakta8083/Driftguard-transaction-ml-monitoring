import { useEffect, useState } from "react";
import { api } from "../api";
import StatusBadge from "../components/StatusBadge";
import PageHeader from "../components/PageHeader";
import KpiCard from "../components/KpiCard";

const PAGE_SIZE = 15;

export default function PredictionHistory() {
  const [page, setPage] = useState(1);
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    api
      .getPredictionHistory(page, PAGE_SIZE)
      .then((res) => !cancelled && setData(res))
      .catch((err) => !cancelled && setError(err.message))
      .finally(() => !cancelled && setLoading(false));
    return () => { cancelled = true; };
  }, [page]);

  const totalPages = data ? Math.max(1, Math.ceil(data.total / PAGE_SIZE)) : 1;

  return (
    <div>
      <PageHeader
        eyebrow="TRANSACTION LOG"
        title="Prediction History"
        subtitle="Every transaction the model has scored so far, most recent first, with its decision and confidence."
      />

      <div className="panel">
        {data && (
          <div className="kpi-row" style={{ marginBottom: 4 }}>
            <KpiCard label="Total scored" value={data.total} />
            <KpiCard label="Flagged as fraud" value={data.total_fraud} tone={data.total_fraud > 0 ? "critical" : "healthy"} />
            <KpiCard label="Legitimate" value={data.total_legitimate} tone="healthy" />
          </div>
        )}
        {error && (
          <div className="error-banner">
            Couldn't load prediction history.
            <span className="hint">
              Make sure the FastAPI server is running at http://localhost:8000. ({error})
            </span>
          </div>
        )}
        {loading && <p className="empty-state">Loading…</p>}
        {!loading && data && data.items.length === 0 && (
          <p className="empty-state">No predictions yet — score a transaction on the Live Scoring page first.</p>
        )}
        {!loading && data && data.items.length > 0 && (
          <>
            <table>
              <thead>
                <tr>
                  <th>Prediction ID</th>
                  <th>Transaction</th>
                  <th>Fraud probability</th>
                  <th>Decision</th>
                  <th>Model version</th>
                  <th>Scored at</th>
                </tr>
              </thead>
              <tbody>
                {data.items.map((item) => (
                  <tr key={item.id}>
                    <td className="mono">{item.id}</td>
                    <td className="mono">#{item.transaction_id}</td>
                    <td className="mono">{(item.fraud_probability * 100).toFixed(2)}%</td>
                    <td>
                      {item.predicted_label === 1 ? (
                        <StatusBadge status="critical" label="Fraud" />
                      ) : (
                        <StatusBadge status="healthy" label="Legitimate" />
                      )}
                    </td>
                    <td>{item.model_version}</td>
                    <td className="mono">{new Date(item.created_at).toLocaleString()}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="pagination">
              <button className="btn-secondary" disabled={page <= 1} onClick={() => setPage((p) => p - 1)}>
                Previous
              </button>
              <span>Page {page} of {totalPages} · {data.total} total predictions</span>
              <button className="btn-secondary" disabled={page >= totalPages} onClick={() => setPage((p) => p + 1)}>
                Next
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

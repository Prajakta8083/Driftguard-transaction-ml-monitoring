import { useEffect, useMemo, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from "recharts";
import { Layers, AlertTriangle } from "lucide-react";
import { api } from "../api";
import KpiCard from "../components/KpiCard";
import StatusBadge from "../components/StatusBadge";
import PageHeader from "../components/PageHeader";

export default function DriftDashboard() {
  const [current, setCurrent] = useState(null);
  const [historyPoints, setHistoryPoints] = useState([]);
  const [selectedFeature, setSelectedFeature] = useState(null);
  const [error, setError] = useState(null);
  const [checking, setChecking] = useState(false);

  async function loadHistory() {
    const hist = await api.getDriftHistory();
    setHistoryPoints(hist.points);
  }

  async function runCheck() {
    setChecking(true);
    setError(null);
    try {
      const res = await api.getDrift();
      setCurrent(res);
      if (!selectedFeature && res.results.length > 0) {
        setSelectedFeature([...res.results].sort((a, b) => b.psi_score - a.psi_score)[0].feature_name);
      }
      await loadHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setChecking(false);
    }
  }

  useEffect(() => {
    runCheck();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sortedResults = useMemo(
    () => (current ? [...current.results].sort((a, b) => b.psi_score - a.psi_score) : []),
    [current]
  );

  const chartData = useMemo(() => {
    if (!selectedFeature) return [];
    return historyPoints
      .filter((p) => p.feature_name === selectedFeature)
      .map((p) => ({ ...p, time: new Date(p.created_at).toLocaleTimeString() }));
  }, [historyPoints, selectedFeature]);

  const overallStatus = current && current.n_drifted >= 3 ? "critical" : current && current.n_drifted > 0 ? "warning" : "healthy";
  const overallLabel = current && current.n_drifted >= 3 ? "Significant drift" : current && current.n_drifted > 0 ? "Minor drift" : "Stable";

  return (
    <div>
      <PageHeader
        eyebrow="DATA DRIFT"
        title="Drift Dashboard"
        subtitle="Compares today's transactions against the data the model was originally trained on. If they look meaningfully different, the model's accuracy may start to slip even before it shows up in performance metrics."
      />

      {error && (
        <div className="error-banner">
          Couldn't run the drift check.
          <span className="hint">Make sure the FastAPI server is running at http://localhost:8000. ({error})</span>
        </div>
      )}

      <div className="kpi-row">
        <KpiCard icon={Layers} label="Features checked" value={current?.n_features_checked ?? "—"} />
        <KpiCard
          icon={AlertTriangle}
          label="Features drifted"
          value={current?.n_drifted ?? "—"}
          tone={current?.n_drifted > 0 ? "critical" : "healthy"}
        />
        <div className="kpi-card">
          <p className="kpi-label">Overall status</p>
          <div style={{ marginTop: 4 }}>
            <StatusBadge status={overallStatus} label={overallLabel} />
          </div>
        </div>
        <div className="kpi-card" style={{ display: "flex", alignItems: "center" }}>
          <button className="btn-secondary" onClick={runCheck} disabled={checking}>
            {checking ? "Checking…" : "Re-check now"}
          </button>
        </div>
      </div>

      <div className="panel-row">
        <div className="panel-col panel">
          <p className="panel-title">Per-feature distribution shift</p>
          <p className="panel-subtext">
            Score shown is PSI (Population Stability Index) — 0 means identical to training data;
            above 0.2 is considered a meaningful shift. Click a feature to plot its trend.
          </p>
          {sortedResults.length === 0 ? (
            <p className="empty-state">Not enough recent transactions yet (need 30+). Score some on Live Scoring.</p>
          ) : (
            <div className="drift-feature-list">
              {sortedResults.map((r) => (
                <div
                  key={r.feature_name}
                  className={`drift-feature-row${selectedFeature === r.feature_name ? " selected" : ""}`}
                  onClick={() => setSelectedFeature(r.feature_name)}
                >
                  <span className="mono" style={{ fontSize: 13, fontWeight: 600 }}>{r.feature_name}</span>
                  <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <span className="mono" style={{ fontSize: 12, color: "var(--text-muted)" }}>
                      PSI {r.psi_score.toFixed(3)}
                    </span>
                    {r.is_drifted ? <StatusBadge status="critical" label="Drifted" /> : <StatusBadge status="healthy" label="Stable" />}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="panel-col panel">
          <p className="panel-title">{selectedFeature ? `${selectedFeature} — shift over time` : "Select a feature"}</p>
          <p className="panel-subtext">The dashed line marks the drift threshold (0.2).</p>
          {chartData.length === 0 ? (
            <p className="empty-state">
              History builds up as drift checks run over time (each check adds one point per feature).
            </p>
          ) : (
            <ResponsiveContainer width="100%" height={240}>
              <LineChart data={chartData}>
                <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
                <XAxis dataKey="time" stroke="var(--text-muted)" fontSize={11} />
                <YAxis stroke="var(--text-muted)" fontSize={11} />
                <Tooltip contentStyle={{ background: "white", border: "1px solid var(--border)" }} />
                <ReferenceLine y={0.2} stroke="var(--status-critical)" strokeDasharray="4 4" label={{ value: "threshold", fill: "var(--status-critical)", fontSize: 10 }} />
                <Line type="monotone" dataKey="drift_score" stroke="#1d4ed8" dot={{ r: 3 }} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}

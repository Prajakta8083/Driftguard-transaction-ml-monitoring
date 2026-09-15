import { useEffect, useState } from "react";
import { Target, Crosshair, Scale, Users } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { api } from "../api";
import KpiCard from "../components/KpiCard";
import StatusBadge from "../components/StatusBadge";
import PageHeader from "../components/PageHeader";

export default function ModelHealth() {
  const [current, setCurrent] = useState(null);
  const [history, setHistory] = useState([]);
  const [error, setError] = useState(null);

  async function load() {
    try {
      const [perf, hist] = await Promise.all([api.getPerformance(), api.getPerformanceHistory()]);
      setCurrent(perf);
      setHistory(hist.points.map((p) => ({ ...p, time: new Date(p.created_at).toLocaleTimeString() })));
      setError(null);
    } catch (err) {
      setError(err.message);
    }
  }

  useEffect(() => {
    load();
    const interval = setInterval(load, 15000);
    return () => clearInterval(interval);
  }, []);

  let status = "healthy";
  let statusLabel = "Healthy";
  if (current && current.n_matched < 5) {
    status = "warning";
    statusLabel = "Not enough data yet";
  } else if (current && current.recall !== null && current.recall < 0.6) {
    status = "critical";
    statusLabel = "Degraded";
  } else if (current && current.recall !== null && current.recall < 0.8) {
    status = "warning";
    statusLabel = "Watch closely";
  }

  return (
    <div>
      <PageHeader
        eyebrow="MODEL PERFORMANCE"
        title="Model Health"
        subtitle="How well the model is actually doing right now, measured against confirmed outcomes — not just how it performed during training."
      />

      {error && (
        <div className="error-banner">
          Couldn't load model health data.
          <span className="hint">Make sure the FastAPI server is running at http://localhost:8000. ({error})</span>
        </div>
      )}

      <div className="kpi-row">
        <KpiCard
          icon={Crosshair}
          label="Precision"
          value={current?.precision != null ? current.precision.toFixed(2) : "—"}
        />
        <KpiCard
          icon={Target}
          label="Recall (fraud caught)"
          value={current?.recall != null ? current.recall.toFixed(2) : "—"}
        />
        <KpiCard
          icon={Scale}
          label="F1 (overall balance)"
          value={current?.f1 != null ? current.f1.toFixed(2) : "—"}
        />
        <KpiCard icon={Users} label="Confirmed outcomes" value={current?.n_matched ?? "—"} />
        <div className="kpi-card">
          <p className="kpi-label">Status</p>
          <div style={{ marginTop: 4 }}>
            <StatusBadge status={status} label={statusLabel} />
          </div>
        </div>
      </div>

      <div className="panel">
        <p className="panel-title">Precision, recall &amp; F1 over time</p>
        <p className="panel-subtext">
          <strong>Precision</strong> = of everything flagged as fraud, how much really was.{" "}
          <strong>Recall</strong> = of all actual fraud, how much was caught.{" "}
          <strong>F1</strong> balances the two into one score.
        </p>
        {history.length === 0 ? (
          <p className="empty-state">
            Not enough confirmed outcomes yet to show a trend. Score some transactions on Live Scoring
            and confirm their outcome — this chart fills in as that history builds up.
          </p>
        ) : (
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={history}>
              <CartesianGrid stroke="var(--border)" strokeDasharray="3 3" />
              <XAxis dataKey="time" stroke="var(--text-muted)" fontSize={11} />
              <YAxis domain={[0, 1]} stroke="var(--text-muted)" fontSize={11} />
              <Tooltip contentStyle={{ background: "white", border: "1px solid var(--border)" }} />
              <Legend />
              <Line type="monotone" dataKey="precision" name="Precision" stroke="#1d4ed8" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="recall" name="Recall" stroke="#16a34a" dot={false} strokeWidth={2} />
              <Line type="monotone" dataKey="f1" name="F1" stroke="#d97706" dot={false} strokeWidth={2} />
            </LineChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}

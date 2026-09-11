import { useEffect, useState } from "react";
import { FileText } from "lucide-react";
import { api } from "../api";
import PageHeader from "../components/PageHeader";
import KpiCard from "../components/KpiCard";

export default function ModelReport() {
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api.getModelReport().then(setReport).catch((err) => setError(err.message));
  }, []);

  const sorted = report?.comparison
    ? [...report.comparison].sort((a, b) => b.pr_auc - a.pr_auc)
    : [];

  return (
    <div>
      <PageHeader
        eyebrow="ANALYSIS REPORT"
        title="Model Report"
        subtitle="Which model is running right now, and the evidence used to pick it — every model that was tried, and how each one scored."
      />

      {error && (
        <div className="error-banner">
          Couldn't load the model report.
          <span className="hint">
            Make sure model_comparison_results.json is in the backend folder, and the server is running. ({error})
          </span>
        </div>
      )}

      {report && (
        <>
          <div className="kpi-row">
            <KpiCard icon={FileText} label="Dataset" value="284,807" unit="transactions" />
            <KpiCard label="Fraud cases" value="492" unit={`(${report.dataset_summary.fraud_rate_pct}%)`} />
            <div className="kpi-card">
              <p className="kpi-label">Model currently running</p>
              <p className="kpi-value" style={{ fontSize: 18 }}>{report.active_model}</p>
            </div>
          </div>

          <div className="panel" style={{ marginBottom: 20 }}>
            <p className="panel-title">What "Model currently running" means</p>
            <p className="panel-subtext" style={{ marginBottom: 0 }}>
              Every prediction on Live Scoring is made by this one model — it was picked automatically
              because it scored the highest PR-AUC (see table below) out of everything tested. If the
              model is ever retrained (Drift Dashboard → Model Health), a new version appears here only
              after someone manually activates it and restarts the server.
            </p>
          </div>

          <div className="panel">
            <p className="panel-title">Every model that was tried</p>
            <p className="panel-subtext">
              Each algorithm was tested two ways — once compensating for the rare fraud cases by making
              mistakes on them cost more ("class-weighted"), and once by generating extra synthetic fraud
              examples to learn from ("SMOTE"). PR-AUC (highlighted) is the metric used to pick the winner
              — see Phase 0 reasoning: with fraud at 0.17% of transactions, precision and recall matter far
              more than raw accuracy.
            </p>
            {sorted.length === 0 ? (
              <p className="empty-state">
                No comparison data found — copy model_comparison_results.json into the backend folder.
              </p>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>PR-AUC</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1</th>
                  </tr>
                </thead>
                <tbody>
                  {sorted.map((row, i) => (
                    <tr key={row.model} style={i === 0 ? { background: "var(--brand-soft)" } : undefined}>
                      <td style={i === 0 ? { fontWeight: 700 } : undefined}>
                        {row.model} {i === 0 && "— winner"}
                      </td>
                      <td className="mono">{row.pr_auc}</td>
                      <td className="mono">{row.precision}</td>
                      <td className="mono">{row.recall}</td>
                      <td className="mono">{row.f1}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </>
      )}
    </div>
  );
}

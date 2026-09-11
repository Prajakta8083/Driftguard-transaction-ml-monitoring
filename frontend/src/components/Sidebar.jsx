import { NavLink } from "react-router-dom";
import { ShieldCheck, Zap, History, HeartPulse, Radar, Eye, FileText, ClipboardList } from "lucide-react";
import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

const NAV_ITEMS = [
  { to: "/", label: "Live Scoring", icon: Zap },
  { to: "/review", label: "Review Queue", icon: ClipboardList },
  { to: "/history", label: "Prediction History", icon: History },
  { to: "/model-health", label: "Model Health", icon: HeartPulse },
  { to: "/drift", label: "Drift Dashboard", icon: Radar },
  { to: "/explainability", label: "Explainability", icon: Eye },
  { to: "/report", label: "Model Report", icon: FileText },
];

export default function Sidebar() {
  const [connected, setConnected] = useState(null); // null=checking, true/false

  useEffect(() => {
    let cancelled = false;
    async function ping() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (!cancelled) setConnected(res.ok);
      } catch {
        if (!cancelled) setConnected(false);
      }
    }
    ping();
    const interval = setInterval(ping, 8000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-mark">
          <ShieldCheck size={18} />
        </div>
        <div className="sidebar-brand-text-group">
          <div className="sidebar-brand-name">DriftGuard</div>
          <div className="sidebar-brand-subtitle">Fraud Monitoring</div>
        </div>
      </div>

      <p className="sidebar-section-label">MONITORING</p>
      <nav className="sidebar-nav">
        {NAV_ITEMS.map((item) => {
          const Icon = item.icon;
          return (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.to === "/"}
              className={({ isActive }) => `sidebar-link${isActive ? " active" : ""}`}
            >
              <Icon size={16} />
              {item.label}
            </NavLink>
          );
        })}
      </nav>

      <div className="sidebar-footer">
        <div className="sidebar-footer-status">
          <span
            className="dot"
            style={{
              background:
                connected === null ? "#94a3b8" : connected ? "var(--status-healthy)" : "var(--status-critical)",
            }}
          />
          {connected === null ? "Checking…" : connected ? "System Online" : "Backend Unreachable"}
        </div>
        <div className="sidebar-footer-sub">
          {connected === false ? "Start the FastAPI server on :8000" : "API Connected"}
        </div>
        <div className="sidebar-footer-meta">
          DriftGuard · Fraud Detection &amp; Model
          <br />
          Health Monitoring Platform
          <br />
          v0.1.0
        </div>
      </div>
    </aside>
  );
}

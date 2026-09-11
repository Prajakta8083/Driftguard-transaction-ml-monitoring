import { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

export default function ApiStatusBadge() {
  // "checking" | "connected" | "offline"
  const [status, setStatus] = useState("checking");

  useEffect(() => {
    let cancelled = false;

    async function ping() {
      try {
        const res = await fetch(`${API_BASE}/health`);
        if (!cancelled) setStatus(res.ok ? "connected" : "offline");
      } catch {
        if (!cancelled) setStatus("offline");
      }
    }

    ping();
    const interval = setInterval(ping, 8000);
    return () => {
      cancelled = true;
      clearInterval(interval);
    };
  }, []);

  const config = {
    checking: { cls: "checking", label: "Checking backend…" },
    connected: { cls: "healthy", label: "API Connected" },
    offline: { cls: "critical", label: "API Offline" },
  }[status];

  return (
    <span className={`api-status-pill ${config.cls}`} title="http://localhost:8000">
      <span className="dot" />
      {config.label}
    </span>
  );
}

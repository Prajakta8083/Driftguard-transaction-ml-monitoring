export default function StatusBadge({ status, label }) {
  // status: "healthy" | "warning" | "critical"
  return (
    <span className={`status-badge ${status}`}>
      <span className="dot" />
      {label}
    </span>
  );
}

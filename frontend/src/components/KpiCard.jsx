export default function KpiCard({ label, value, unit, tone, icon: Icon }) {
  return (
    <div className="kpi-card">
      {Icon && (
        <div className="kpi-icon-box">
          <Icon size={17} />
        </div>
      )}
      <p className="kpi-label">{label}</p>
      <p className={`kpi-value${tone ? ` ${tone}` : ""}`}>
        {value}
        {unit && <span className="kpi-unit">{unit}</span>}
      </p>
    </div>
  );
}

import ApiStatusBadge from "./ApiStatusBadge";

export default function PageHeader({ eyebrow, title, subtitle }) {
  return (
    <div className="page-header-row">
      <div>
        {eyebrow && <p className="page-eyebrow">{eyebrow}</p>}
        <h1 className="page-title">{title}</h1>
        {subtitle && <p className="page-subtitle">{subtitle}</p>}
      </div>
      <ApiStatusBadge />
    </div>
  );
}

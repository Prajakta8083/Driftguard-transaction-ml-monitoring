// Only Time and Amount are human-interpretable in this dataset — V1..V28
// are anonymized PCA components (see Phase 0/2 discussion). We name Time
// and Amount plainly when they show up, and refer to V-features honestly
// as "internal risk factor VN" rather than inventing a false meaning for
// them — overstating interpretability here would be misleading.
const FRIENDLY_NAMES = {
  Time: "transaction timing",
  Amount: "transaction amount",
};

function friendlyName(feature) {
  return FRIENDLY_NAMES[feature] || `internal risk factor ${feature}`;
}

export function buildPlainExplanation(topDrivers, predictedLabel) {
  if (!topDrivers || topDrivers.length === 0) {
    return "No explanation available for this prediction yet.";
  }

  // Drivers pushing TOWARD the model's actual decision, sorted by magnitude.
  const sorted = [...topDrivers].sort(
    (a, b) => Math.abs(b.shap_value) - Math.abs(a.shap_value)
  );
  const relevant =
    predictedLabel === 1
      ? sorted.filter((d) => d.shap_value > 0)
      : sorted.filter((d) => d.shap_value < 0);

  const top = (relevant.length > 0 ? relevant : sorted).slice(0, 2);
  const names = top.map((d) => friendlyName(d.feature));
  const joined =
    names.length === 1 ? names[0] : `${names[0]} and ${names[1]}`;

  return predictedLabel === 1
    ? `Flagged mainly due to unusual ${joined}.`
    : `Scored as low risk — ${joined} looked consistent with normal transactions.`;
}

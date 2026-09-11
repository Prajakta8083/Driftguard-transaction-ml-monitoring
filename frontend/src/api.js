const API_BASE = "http://localhost:8000";

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json();
}

export const api = {
  postTransaction: (payload) =>
    request("/transactions", { method: "POST", body: JSON.stringify(payload) }),

  getPredictionHistory: (page = 1, pageSize = 20) =>
    request(`/predictions/history?page=${page}&page_size=${pageSize}`),
  getReviewQueue: () => request("/predictions/review-queue"),

  postFeedback: (transactionId, actualLabel) =>
    request("/feedback", {
      method: "POST",
      body: JSON.stringify({ transaction_id: transactionId, actual_label: actualLabel }),
    }),

  getPerformance: () => request("/monitoring/performance"),
  getPerformanceHistory: (limit = 200) =>
    request(`/monitoring/performance/history?limit=${limit}`),

  getDrift: () => request("/monitoring/drift"),
  getDriftHistory: (limit = 500) => request(`/monitoring/drift/history?limit=${limit}`),

  explainPrediction: (transactionId) => request(`/model/explain/${transactionId}`),
  getGlobalImportance: () => request("/model/global-importance"),
  getModelReport: () => request("/model/report"),

  retrainCheck: () => request("/model/retrain-check", { method: "POST" }),
};

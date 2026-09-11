import { Routes, Route } from "react-router-dom";
import Sidebar from "./components/Sidebar";
import LiveScoring from "./pages/LiveScoring";
import PredictionHistory from "./pages/PredictionHistory";
import ModelHealth from "./pages/ModelHealth";
import DriftDashboard from "./pages/DriftDashboard";
import Explainability from "./pages/Explainability";
import ModelReport from "./pages/ModelReport";
import ReviewQueue from "./pages/ReviewQueue";

export default function App() {
  return (
    <div className="app-shell">
      <Sidebar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<LiveScoring />} />
          <Route path="/history" element={<PredictionHistory />} />
          <Route path="/model-health" element={<ModelHealth />} />
          <Route path="/drift" element={<DriftDashboard />} />
          <Route path="/explainability" element={<Explainability />} />
          <Route path="/report" element={<ModelReport />} />
          <Route path="/review" element={<ReviewQueue />} />
        </Routes>
      </main>
    </div>
  );
}

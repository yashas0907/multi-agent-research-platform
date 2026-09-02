import { useState } from "react";
import Workspace from "./components/Workspace";
import ResearchView from "./components/ResearchView";

export default function App() {
  const [sessionId, setSessionId] = useState<string | null>(null);

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">RI</div>
          <div>
            <h1>Research &amp; Intelligence Platform</h1>
            <small>Multi-agent research pipeline with source verification</small>
          </div>
        </div>
      </header>

      {!sessionId ? (
        <Workspace onStart={setSessionId} />
      ) : (
        <ResearchView sessionId={sessionId} onReset={() => setSessionId(null)} />
      )}
    </div>
  );
}

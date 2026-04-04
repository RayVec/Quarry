import type { SessionState } from "../types";

interface DiagnosticsDrawerProps {
  session: SessionState | null;
  open: boolean;
  onClose: () => void;
}

export function DiagnosticsDrawer({ session, open, onClose }: DiagnosticsDrawerProps) {
  if (!open) {
    return null;
  }

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside className="diagnostics-drawer" data-testid="diagnostics-drawer" onClick={(event) => event.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <span className="eyebrow">Developer diagnostics</span>
            <h2>Runtime details</h2>
          </div>
          <button className="icon-button" data-testid="close-diagnostics" onClick={onClose}>
            Close
          </button>
        </div>

        {session ? (
          <div className="drawer-stack">
            <section className="drawer-section">
              <span className="tiny-label">Session</span>
              <p>{session.session_id}</p>
              <p data-testid="status-mode">Mode: {session.response_mode}</p>
              <p data-testid="status-query-status">Query status: {session.query_status}</p>
              <p data-testid="status-query-stage">Current stage: {session.query_stage_label}</p>
              <p data-testid="status-runtime-mode">Runtime: {session.runtime_mode}</p>
              <p data-testid="status-runtime-profile">Profile: {session.runtime_profile}</p>
              <p data-testid="status-generation-provider">Generator: {session.generation_provider}</p>
              <p data-testid="status-parser-provider">Parser: {session.parser_provider}</p>
              <p data-testid="status-refinements">Refinements: {session.refinement_count}</p>
              <p data-testid="status-citations">Citations: {session.citation_index.length}</p>
              <p data-testid="status-sentences">Sentences: {session.parsed_sentences.length}</p>
            </section>

            {session.ui_messages.length ? (
              <section className="drawer-section">
                <span className="tiny-label">Pipeline messages</span>
                {session.ui_messages.map((message) => (
                  <div className={`message-row ${message.level}`} key={`${message.code}-${message.message}`}>
                    <strong>{message.code}</strong>
                    <p>{message.message}</p>
                  </div>
                ))}
              </section>
            ) : null}

            {session.retrieval_diagnostics.length ? (
              <section className="drawer-section">
                <span className="tiny-label">Retrieval diagnostics</span>
                {session.retrieval_diagnostics.map((diagnostic) => (
                  <div className="diagnostic-row" key={diagnostic.facet}>
                    <strong>{diagnostic.facet}</strong>
                    <p>
                      sparse {diagnostic.sparse.result_count} · dense {diagnostic.dense.result_count} · fused {diagnostic.fused_count} · reranked {diagnostic.reranked_count}
                    </p>
                    {diagnostic.degraded_mode ? <p>degraded fallback in use</p> : null}
                  </div>
                ))}
              </section>
            ) : null}

            <section className="drawer-section">
              <span className="tiny-label">Local model status</span>
              {Object.entries(session.local_model_status).map(([key, value]) => (
                <div className="diagnostic-row" key={key}>
                  <strong>{key}</strong>
                  <p>{value}</p>
                </div>
              ))}
            </section>
          </div>
        ) : (
          <div className="drawer-section">
            <span className="tiny-label">Status</span>
            <p>No session is active yet.</p>
          </div>
        )}
      </aside>
    </div>
  );
}

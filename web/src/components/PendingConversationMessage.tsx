import type { QueryProgressStage, QueryRunStatus, SessionState } from "../types";

interface PendingConversationMessageProps {
  session: SessionState | null;
}

interface PendingStage {
  key: QueryProgressStage;
  title: string;
  detail: string;
}

const PENDING_STAGES: PendingStage[] = [
  {
    key: "understanding",
    title: "Reading your question",
    detail: "I'm getting clear on what you want to know.",
  },
  {
    key: "searching",
    title: "Looking through the reports",
    detail: "I'm finding the parts of the documents that seem most relevant.",
  },
  {
    key: "evidence",
    title: "Pulling together the best evidence",
    detail: "I'm narrowing down to the passages I trust most for this answer.",
  },
  {
    key: "writing",
    title: "Writing the answer",
    detail: "I'm turning the evidence into a clear response.",
  },
  {
    key: "checking",
    title: "Checking the answer against the reports",
    detail: "I'm making sure the wording still matches the source text before I show it.",
  },
];

const STAGE_ORDER = new Map(PENDING_STAGES.map((stage, index) => [stage.key, index]));

function fallbackStage() {
  return {
    status: "running" as QueryRunStatus,
    stage: "queued" as QueryProgressStage,
  };
}

export function PendingConversationMessage({ session }: PendingConversationMessageProps) {
  const stageState = session
    ? {
        status: session.query_status,
        stage: session.query_stage,
      }
    : fallbackStage();
  const stageIndex = STAGE_ORDER.get(stageState.stage) ?? -1;
  const isFailed = stageState.status === "failed";

  return (
    <article
      className={`thread-message assistant-message pending ${isFailed ? "failed" : ""}`}
      data-testid="pending-response"
    >
      <div className="thread-message-header">
        <div>
          <span className="eyebrow">QUARRY response</span>
        </div>
        {isFailed ? <span className="pending-status-chip">Needs retry</span> : null}
      </div>

      <div className="pending-stage-list" aria-live="polite">
        {PENDING_STAGES.map((stage, index) => {
          const status =
            stageIndex < 0 ? "upcoming" : index < stageIndex ? "done" : index === stageIndex ? "active" : "upcoming";
          return (
            <div className={`pending-stage ${status}`} key={stage.key}>
              <span className={`pending-stage-marker ${status}`} aria-hidden="true" />
              <div>
                <strong>{stage.title}</strong>
                <p>{stage.detail}</p>
              </div>
            </div>
          );
        })}
      </div>
    </article>
  );
}

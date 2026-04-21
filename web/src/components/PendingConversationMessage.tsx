import { Badge } from "@/components/ui/badge";
import { CardContent, CardHeader } from "@/components/ui/card";
import type {
  MessageRunState,
  QueryProgressStage,
  QueryRunStatus,
  QueryStageDescriptor,
  SessionState,
} from "../types";

interface PendingConversationMessageProps {
  messageRun: MessageRunState | null;
  session: SessionState | null;
}

type PendingStageStatus = "done" | "active" | "upcoming";
type PendingProgressState =
  | Pick<SessionState, "query_status" | "query_stage" | "query_stage_label" | "query_stage_detail" | "query_stage_catalog">
  | Pick<MessageRunState, "status" | "stage" | "stage_label" | "stage_detail" | "stage_catalog">;

export const DEFAULT_PENDING_STAGES: QueryStageDescriptor[] = [
  {
    key: "orchestrating",
    label: "Deciding whether to search",
    detail: "I'm deciding whether this needs report search or a direct response.",
  },
  {
    key: "understanding",
    label: "Reading your question",
    detail: "I'm getting clear on what you want to know.",
  },
  {
    key: "searching",
    label: "Looking through the reports",
    detail: "I'm finding the parts of the documents that seem most relevant.",
  },
  {
    key: "evidence",
    label: "Pulling together the best evidence",
    detail: "I'm narrowing down to the passages I trust most for this answer.",
  },
  {
    key: "coverage_check",
    label: "Checking evidence coverage",
    detail: "I'm checking whether each facet is supported by cited evidence.",
  },
  {
    key: "writing",
    label: "Writing the answer",
    detail: "I'm turning the evidence into a clear response.",
  },
  {
    key: "checking",
    label: "Checking the answer against the reports",
    detail: "I'm making sure the wording still matches the source text before I show it.",
  },
];

function fallbackStageState() {
  return {
    status: "running" as QueryRunStatus,
    stage: "orchestrating" as QueryProgressStage,
    stage_label: "Deciding whether to search",
    stage_detail: "I'm deciding whether this needs report search or a direct response.",
    stage_catalog: DEFAULT_PENDING_STAGES,
  };
}

function toDisplayStageKey(stage: QueryProgressStage): QueryProgressStage {
  if (stage === "orchestrating") {
    return "understanding";
  }
  if (stage === "followup_retrieval") {
    return "coverage_check";
  }
  return stage;
}

function isSessionProgressState(
  progressState: PendingProgressState,
): progressState is Pick<
  SessionState,
  "query_status" | "query_stage" | "query_stage_label" | "query_stage_detail" | "query_stage_catalog"
> {
  return "query_stage_catalog" in progressState;
}

function resolvePendingProgressState(
  messageRun: MessageRunState | null,
  session: SessionState | null,
): PendingProgressState {
  if (session) {
    return {
      query_status: session.query_status,
      query_stage: session.query_stage,
      query_stage_label: session.query_stage_label,
      query_stage_detail: session.query_stage_detail,
      query_stage_catalog: session.query_stage_catalog ?? DEFAULT_PENDING_STAGES,
    };
  }
  if (messageRun) {
    return messageRun;
  }
  return fallbackStageState();
}

export function getPendingStages(progressState: PendingProgressState): QueryStageDescriptor[] {
  let catalog: QueryStageDescriptor[] | undefined;
  if (isSessionProgressState(progressState)) {
    catalog = progressState.query_stage_catalog;
  } else {
    catalog = progressState.stage_catalog;
  }

  if (catalog?.length) {
    const collapsed = new Map<QueryProgressStage, QueryStageDescriptor>();
    for (const stage of catalog) {
      const displayKey = toDisplayStageKey(stage.key);
      const existing = collapsed.get(displayKey);
      if (existing && displayKey !== stage.key) {
        continue;
      }
      collapsed.set(
        displayKey,
        displayKey === stage.key ? stage : { ...stage, key: displayKey },
      );
    }
    return [...collapsed.values()];
  }
  return DEFAULT_PENDING_STAGES;
}

export function getPendingStageStatuses(
  status: QueryRunStatus,
  stage: QueryProgressStage,
  stages: QueryStageDescriptor[] = DEFAULT_PENDING_STAGES,
): PendingStageStatus[] {
  const stageOrder = new Map(stages.map((item, index) => [item.key, index]));
  const displayStage = toDisplayStageKey(stage);
  const stageIndex =
    stage === "completed"
      ? stages.length - 1
      : (stageOrder.get(displayStage) ?? 0);

  if (status === "completed") {
    return stages.map((_, index) =>
      index <= stageIndex ? "done" : "upcoming",
    );
  }

  return stages.map((_, index) => {
    if (index < stageIndex) {
      return "done";
    }
    if (index === stageIndex) {
      return "active";
    }
    return "upcoming";
  });
}

export function getDisplayedPendingStages(
  progressState: PendingProgressState,
): QueryStageDescriptor[] {
  const stages = getPendingStages(progressState);
  const rawActiveStage =
    "query_stage" in progressState ? progressState.query_stage : progressState.stage;
  const activeStage = toDisplayStageKey(rawActiveStage);
  const activeLabel =
    "query_stage_label" in progressState ? progressState.query_stage_label : progressState.stage_label;
  const activeDetail =
    "query_stage_detail" in progressState ? progressState.query_stage_detail : progressState.stage_detail;
  const shouldUseActiveCopy = rawActiveStage === activeStage;

  return stages.map((stage) =>
    stage.key === activeStage
      ? {
          ...stage,
          label: shouldUseActiveCopy ? (activeLabel || stage.label) : stage.label,
          detail: shouldUseActiveCopy ? (activeDetail || stage.detail) : stage.detail,
        }
      : stage,
  );
}

export function PendingConversationMessage({
  messageRun,
  session,
}: PendingConversationMessageProps) {
  const progressState = resolvePendingProgressState(messageRun, session);
  const displayedStages = getDisplayedPendingStages(progressState);
  const stageState = "query_status" in progressState
    ? {
        status: progressState.query_status,
        stage: progressState.query_stage,
      }
    : {
        status: progressState.status,
        stage: progressState.stage,
      };
  const stageStatuses = getPendingStageStatuses(
    stageState.status,
    stageState.stage,
    displayedStages,
  );
  const isFailed = stageState.status === "failed";

  return (
    <section
      className={`thread-message assistant-message pending ${isFailed ? "failed" : ""}`}
      data-testid="pending-response"
    >
      <CardHeader className="thread-message-header">
        <div>
          <span className="eyebrow">QUARRY response</span>
        </div>
        {isFailed ? <Badge className="pending-status-chip">Needs retry</Badge> : null}
      </CardHeader>

      <CardContent className="pending-stage-list" aria-live="polite">
        {displayedStages.map((stage, index) => {
          const status = stageStatuses[index];
          return (
            <div className={`pending-stage ${status}`} key={stage.key}>
              <span className={`pending-stage-marker ${status}`} aria-hidden="true" />
              <div>
                <strong>{stage.label}</strong>
                <p>{stage.detail}</p>
              </div>
            </div>
          );
        })}
      </CardContent>
    </section>
  );
}

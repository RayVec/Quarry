import { describe, expect, it } from "vitest";
import type { SessionState } from "../types";
import {
  getDisplayedPendingStages,
  getPendingStageStatuses,
} from "./PendingConversationMessage";

describe("PendingConversationMessage stage mapping", () => {
  it("shows orchestration as the first active backend stage", () => {
    expect(getPendingStageStatuses("running", "orchestrating").slice(0, 3)).toEqual([
      "done",
      "active",
      "upcoming",
    ]);
  });

  it("marks orchestration done once understanding begins", () => {
    expect(getPendingStageStatuses("running", "understanding").slice(0, 3)).toEqual([
      "done",
      "active",
      "upcoming",
    ]);
  });

  it("maps follow-up retrieval onto the shared evidence coverage step", () => {
    expect(getPendingStageStatuses("running", "followup_retrieval")).toEqual([
      "done",
      "done",
      "done",
      "done",
      "active",
      "upcoming",
      "upcoming",
    ]);
  });

  it("uses backend-provided label and detail for the active stage", () => {
    const stages = getDisplayedPendingStages({
      query_status: "running",
      query_stage: "searching",
      query_stage_label: "Inspecting the indexed reports",
      query_stage_detail: "Backend supplied detail for the current step.",
      query_stage_catalog: [
        {
          key: "orchestrating",
          label: "Deciding whether to search",
          detail: "Default orchestration detail.",
        },
        {
          key: "understanding",
          label: "Reading your question",
          detail: "Default understanding detail.",
        },
        {
          key: "searching",
          label: "Looking through the reports",
          detail: "Default retrieval detail.",
        },
      ],
    } as SessionState);

    expect(stages).toEqual([
      {
        key: "understanding",
        label: "Reading your question",
        detail: "Default understanding detail.",
      },
      {
        key: "searching",
        label: "Inspecting the indexed reports",
        detail: "Backend supplied detail for the current step.",
      },
    ]);
  });

  it("keeps follow-up retrieval on the shared coverage copy instead of showing a new label", () => {
    const stages = getDisplayedPendingStages({
      query_status: "running",
      query_stage: "followup_retrieval",
      query_stage_label: "Retrieving additional evidence",
      query_stage_detail: "Backend follow-up detail.",
      query_stage_catalog: [
        {
          key: "coverage_check",
          label: "Checking evidence coverage",
          detail: "Coverage detail.",
        },
        {
          key: "followup_retrieval",
          label: "Retrieving additional evidence",
          detail: "Follow-up detail.",
        },
      ],
    } as SessionState);

    expect(stages).toEqual([
      {
        key: "coverage_check",
        label: "Checking evidence coverage",
        detail: "Coverage detail.",
      },
    ]);
  });

  it("keeps orchestration on the shared reading copy instead of showing a new label", () => {
    const stages = getDisplayedPendingStages({
      status: "running",
      stage: "orchestrating",
      stage_label: "Deciding whether to search",
      stage_detail: "Backend orchestration detail.",
      stage_catalog: [
        {
          key: "orchestrating",
          label: "Deciding whether to search",
          detail: "Orchestration detail.",
        },
        {
          key: "understanding",
          label: "Reading your question",
          detail: "Understanding detail.",
        },
      ],
    });

    expect(stages).toEqual([
      {
        key: "understanding",
        label: "Reading your question",
        detail: "Understanding detail.",
      },
    ]);
  });
});

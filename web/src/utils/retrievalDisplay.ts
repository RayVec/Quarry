import type { Reference, SentenceStatus } from "../types";

/**
 * Same threshold as backend `ambiguity_gap_threshold` (see quarry.config).
 * When the gap between the top two retrieval scores is below this, the run is "tight race".
 */
const AMBIGUITY_GAP_THRESHOLD = 0.05;

/** Clear lead between #1 and #2 hit (same units as retrieval scores, typically 0–1). */
const CLEAR_LEAD_GAP = 0.05;

export type UnifiedMatchLevel = "strong" | "good" | "fair" | "weak";

export interface UnifiedMatchQuality {
  level: UnifiedMatchLevel;
  headline: string;
  detail: string;
}

interface MatchQualityContext {
  sentenceStatus?: SentenceStatus | null;
  referenceVerified?: boolean;
  referenceConfidenceLabel?: Reference["confidence_label"];
  referenceConfidenceUnknown?: boolean;
}

/**
 * Single user-facing match quality from technical signals:
 *
 * Strategy:
 * 1. Treat `retrieval_score` as a rough ranking signal, not calibrated confidence.
 * 2. Use broad score bands only.
 * 3. Use a clear lead to keep the top retrieval result out of `strong` when the runner-up is very close.
 * 4. Mix in verification: partial support caps the result at `fair`, unknown support caps it at `good`,
 *    and unsupported sentences/references fall to `weak`.
 *
 * Labels:
 * - strong: high retrieval score and direct verification support
 * - good: relevant and verified enough to trust, but not at the top band
 * - fair: relevant passage, but support for the current sentence is only partial
 * - weak: poor, unverified, or unsupported match
 */
export function describeUnifiedMatchQuality(
  retrievalScore: number,
  ambiguityReviewRequired: boolean,
  ambiguityGap: number | null | undefined,
  context: MatchQualityContext = {},
): UnifiedMatchQuality {
  if (!Number.isFinite(retrievalScore)) {
    return {
      level: "weak",
      headline: "Match quality unknown",
      detail: "No retrieval score was available.",
    };
  }

  const tightRace =
    ambiguityReviewRequired ||
    (ambiguityGap != null && Number.isFinite(ambiguityGap) && ambiguityGap < AMBIGUITY_GAP_THRESHOLD);
  const hasClearLead = ambiguityGap != null && Number.isFinite(ambiguityGap) && ambiguityGap >= CLEAR_LEAD_GAP;
  const hasComparableRunnerUp = ambiguityGap != null && Number.isFinite(ambiguityGap);

  let scoreTier: 1 | 2 | 3 | 4;
  if (retrievalScore >= 0.9) scoreTier = 4;
  else if (retrievalScore >= 0.72) scoreTier = 3;
  else if (retrievalScore >= 0.5) scoreTier = 2;
  else scoreTier = 1;

  if (scoreTier === 4 && hasComparableRunnerUp && !hasClearLead) {
    scoreTier = 3;
  }

  let finalTier = (tightRace ? Math.max(1, scoreTier - 1) : scoreTier) as 1 | 2 | 3 | 4;
  let detailReason: "retrieval" | "supported" | "partial" | "unknown" | "unsupported" =
    finalTier >= 3 ? "supported" : "retrieval";

  const hasVerificationContext =
    context.sentenceStatus != null ||
    context.referenceVerified != null ||
    context.referenceConfidenceLabel != null ||
    Boolean(context.referenceConfidenceUnknown);
  const sentenceUnsupported =
    context.sentenceStatus === "ungrounded" || context.sentenceStatus === "no_ref";
  const sentencePartiallyVerified = context.sentenceStatus === "partially_verified";
  const sentenceUnchecked = context.sentenceStatus === "unchecked";
  const referenceUnsupported =
    context.referenceVerified === false || context.referenceConfidenceLabel === "not_supported";
  const referencePartial = context.referenceConfidenceLabel === "partially_supported";
  const referenceUnknown =
    hasVerificationContext &&
    (context.referenceConfidenceUnknown || context.referenceConfidenceLabel == null);

  if (sentenceUnsupported || referenceUnsupported) {
    finalTier = 1;
    detailReason = "unsupported";
  } else if (context.referenceVerified === true || context.sentenceStatus === "verified") {
    // Verified evidence should not collapse to weak solely because retrieval_score is low
    // (common with rank-fusion scoring), even when a citation-to-reference link is missing.
    if (context.referenceConfidenceLabel === "supported" || context.sentenceStatus === "verified") {
      finalTier = Math.max(finalTier, tightRace ? 2 : 3) as 1 | 2 | 3 | 4;
      detailReason = tightRace ? "unknown" : "supported";
    } else if (context.referenceConfidenceLabel === "partially_supported" || sentencePartiallyVerified) {
      finalTier = Math.max(finalTier, 2) as 1 | 2 | 3 | 4;
      detailReason = "partial";
    } else {
      finalTier = Math.max(finalTier, 2) as 1 | 2 | 3 | 4;
      detailReason = "unknown";
    }
  } else if (sentencePartiallyVerified || referencePartial) {
    finalTier = Math.min(finalTier, 2) as 1 | 2;
    detailReason = "partial";
  } else if (sentenceUnchecked || referenceUnknown) {
    finalTier = Math.min(finalTier, 3) as 1 | 2 | 3;
    detailReason = "unknown";
  }

  const levelByTier: Record<1 | 2 | 3 | 4, UnifiedMatchLevel> = {
    4: "strong",
    3: "good",
    2: "fair",
    1: "weak",
  };
  const headlineByTier: Record<1 | 2 | 3 | 4, string> = {
    4: "Strong match",
    3: "Good match",
    2: "Fair match",
    1: "Weak match",
  };

  let detail: string;
  switch (finalTier) {
    case 4:
      detail = "This passage clearly supports the sentence.";
      break;
    case 3:
      detail =
        detailReason === "unknown"
          ? "This passage looks relevant, but support could not be confirmed."
          : "This passage supports the sentence.";
      break;
    case 2:
      detail =
        detailReason === "partial"
          ? "This passage supports only part of the sentence."
          : "This passage may be relevant, but the match is not clear.";
      break;
    default:
      detail =
        detailReason === "unsupported"
          ? "This passage does not support the sentence."
          : "This passage is weak or uncertain.";
  }

  return {
    level: levelByTier[finalTier],
    headline: headlineByTier[finalTier],
    detail,
  };
}

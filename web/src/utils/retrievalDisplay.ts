/**
 * Same threshold as backend `ambiguity_gap_threshold` (see quarry.config).
 * When the gap between the top two retrieval scores is below this, the run is "tight race".
 */
const AMBIGUITY_GAP_THRESHOLD = 0.05;

/** Clear lead between #1 and #2 hit (same units as retrieval scores, typically 0–1). */
const CLEAR_LEAD_GAP = 0.12;

export type UnifiedMatchLevel = "strong" | "good" | "fair" | "weak";

export interface UnifiedMatchQuality {
  level: UnifiedMatchLevel;
  headline: string;
  detail: string;
}

/**
 * Single user-facing match quality from technical signals:
 *
 * Strategy:
 * 1. Treat `retrieval_score` as a rough ranking signal, not calibrated confidence.
 * 2. Use broad score bands only.
 * 3. Require a non-tight race for the highest label.
 * 4. Downgrade one level when the top two passages are too close together.
 *
 * Labels:
 * - strong: high score and clear lead
 * - good: high/reasonable score, but not decisive enough for strong
 * - fair: somewhat relevant, needs checking
 * - weak: poor or uncertain match
 */
export function describeUnifiedMatchQuality(
  retrievalScore: number,
  ambiguityReviewRequired: boolean,
  ambiguityGap: number | null | undefined,
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

  if (scoreTier === 4 && (!hasComparableRunnerUp || !hasClearLead)) {
    scoreTier = 3;
  }

  const finalTier = (tightRace ? Math.max(1, scoreTier - 1) : scoreTier) as 1 | 2 | 3 | 4;
  const downgraded = tightRace && scoreTier > finalTier;

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
      detail = "Highly relevant and well aligned.";
      break;
    case 3:
      detail = "Relevant and reasonably aligned.";
      break;
    case 2:
      detail = "Somewhat relevant, but less clear.";
      break;
    default:
      detail = "Low relevance or uncertain support.";
  }

  if (downgraded) {
    detail += " Another passage was nearly as strong.";
  } else if (!hasComparableRunnerUp && finalTier >= 3) {
    detail += " No runner-up was available.";
  } else if (!tightRace && hasClearLead && finalTier >= 3) {
    detail += " It clearly led the next result.";
  }

  return {
    level: levelByTier[finalTier],
    headline: headlineByTier[finalTier],
    detail,
  };
}

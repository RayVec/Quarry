import { useEffect } from "react";
import { persistRecentResearch, persistThread } from "./storage";
import type { RecentResearchItem, ThreadEntry } from "./model";

export function usePersistentThread(
  thread: ThreadEntry[],
  recentResearch: RecentResearchItem[],
) {
  useEffect(() => {
    persistThread(thread);
  }, [thread]);

  useEffect(() => {
    persistRecentResearch(recentResearch);
  }, [recentResearch]);
}

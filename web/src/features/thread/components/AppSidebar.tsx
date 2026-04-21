import { Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  formatRecentResearchDate,
  type RecentResearchItem,
} from "@/features/thread/model";

interface AppSidebarProps {
  recentResearch: RecentResearchItem[];
  threadTitleQuery: string | null;
  onDeleteRecentResearch: (itemId: string) => void;
  onNewSearch: () => void;
  onResumeResearch: (query: string) => Promise<void>;
}

export function AppSidebar({
  recentResearch,
  threadTitleQuery,
  onDeleteRecentResearch,
  onNewSearch,
  onResumeResearch,
}: AppSidebarProps) {
  return (
    <aside className="app-sidebar">
      <div className="sidebar-brand">
        <span className="sidebar-mark">Q</span>
        <div>
          <p className="sidebar-title">QUARRY</p>
          <p className="sidebar-subtitle">Architectural Intelligence</p>
        </div>
      </div>

      <Button
        className="sidebar-primary-action sidebar-primary-action--new-search"
        onClick={onNewSearch}
      >
        + New Search
      </Button>

      <div className="sidebar-section">
        <span className="tiny-label">Recent Research</span>
        {recentResearch.length ? (
          <div className="recent-research-list">
            {recentResearch.map((item, index) => (
              <div
                className={`recent-research-item ${index === 0 && threadTitleQuery === item.query ? "active" : ""}`}
                key={item.id}
              >
                <Button
                  className="recent-research-open recent-research-open--multiline"
                  onClick={() => void onResumeResearch(item.query)}
                  variant="ghost"
                >
                  <strong>{item.query}</strong>
                  <span>{formatRecentResearchDate(item.createdAt)}</span>
                </Button>
                <Button
                  className="recent-research-delete"
                  aria-label={`Delete search ${item.query}`}
                  onClick={() => onDeleteRecentResearch(item.id)}
                >
                  <Trash2 aria-hidden="true" focusable="false" />
                </Button>
              </div>
            ))}
          </div>
        ) : (
          <p className="sidebar-empty">
            Your recent research questions will appear here so you can return
            to them quickly.
          </p>
        )}
      </div>
    </aside>
  );
}

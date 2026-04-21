import { Cog } from "lucide-react";
import { Button } from "@/components/ui/button";
import { QueryComposer } from "@/components/QueryComposer";

interface LandingStageProps {
  loading: boolean;
  query: string;
  onChangeQuery: (nextQuery: string) => void;
  onOpenSettings: () => void;
  onSubmitQuery: () => void;
}

export function LandingStage({
  loading,
  query,
  onChangeQuery,
  onOpenSettings,
  onSubmitQuery,
}: LandingStageProps) {
  return (
    <main className="landing-stage">
      <div className="landing-utility-row">
        <Button
          className="diagnostics-trigger"
          data-testid="open-workspace-settings"
          onClick={onOpenSettings}
        >
          <span className="sr-only">Provider settings</span>
          <Cog aria-hidden="true" focusable="false" />
        </Button>
      </div>

      <section className="hero-block">
        <h1>Intelligence for the built environment.</h1>
        <p>
          Ask complex questions, analyze technical reports, and verify
          construction guidance with editorial clarity and structural
          precision.
        </p>
      </section>

      <QueryComposer
        className="landing"
        id="query-input"
        loading={loading}
        placeholder="Ask a question or reply"
        query={query}
        onChange={onChangeQuery}
        onSubmit={onSubmitQuery}
      />
    </main>
  );
}

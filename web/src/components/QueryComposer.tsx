import { useEffect, useRef, useState, type Ref } from "react";

interface QueryComposerProps {
  id: string;
  label?: string;
  placeholder: string;
  query: string;
  loading: boolean;
  className: string;
  formRef?: Ref<HTMLFormElement>;
  onChange: (nextQuery: string) => void;
  onSubmit: () => void;
}

const MAX_QUERY_LINES = 7;

function measureTextarea(textarea: HTMLTextAreaElement) {
  const computed = window.getComputedStyle(textarea);
  const lineHeight = Number.parseFloat(computed.lineHeight) || 28;
  const verticalPadding =
    Number.parseFloat(computed.paddingTop) + Number.parseFloat(computed.paddingBottom);
  const border =
    Number.parseFloat(computed.borderTopWidth) + Number.parseFloat(computed.borderBottomWidth);
  const minHeight = Math.ceil(lineHeight + verticalPadding + border);
  const maxHeight = Math.ceil(lineHeight * MAX_QUERY_LINES + verticalPadding + border);
  return { minHeight, maxHeight };
}

export function QueryComposer({
  id,
  label,
  placeholder,
  query,
  loading,
  className,
  formRef,
  onChange,
  onSubmit,
}: QueryComposerProps) {
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [multiline, setMultiline] = useState(false);
  const canSubmit = query.trim().length > 0 && !loading;

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) {
      return;
    }

    const { minHeight, maxHeight } = measureTextarea(textarea);
    if (query.length === 0) {
      textarea.style.height = `${minHeight}px`;
      textarea.style.overflowY = "hidden";
      setMultiline(false);
      return;
    }

    textarea.style.height = "0px";
    const nextHeight = Math.min(Math.max(textarea.scrollHeight, minHeight), maxHeight);
    textarea.style.height = `${nextHeight}px`;
    textarea.style.overflowY = textarea.scrollHeight > maxHeight ? "auto" : "hidden";
    setMultiline(nextHeight > minHeight + 2);
  }, [query]);

  return (
    <form
      className={`query-composer ${className}`}
      ref={formRef}
      onSubmit={(event) => {
        event.preventDefault();
        if (!canSubmit) {
          return;
        }
        onSubmit();
      }}
    >
      {label ? (
        <label className="tiny-label" htmlFor={id}>
          {label}
        </label>
      ) : null}
      <div className={`composer-surface ${multiline ? "multiline" : "singleline"}`}>
        <textarea
          id={id}
          className="composer-textarea"
          data-testid="query-input"
          placeholder={placeholder}
          ref={textareaRef}
          rows={1}
          value={query}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={(event) => {
            if (event.key !== "Enter" || event.shiftKey) {
              return;
            }
            event.preventDefault();
            if (!canSubmit) {
              return;
            }
            onSubmit();
          }}
        />
        <div className="composer-submit-row">
          <button
            className="primary-button composer-submit"
            data-testid="run-query"
            disabled={!canSubmit}
            type="submit"
          >
            {loading ? "Working..." : "Ask QUARRY"}
          </button>
        </div>
      </div>
    </form>
  );
}

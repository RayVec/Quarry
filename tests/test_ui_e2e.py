from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import time
from urllib.error import URLError
from urllib.request import urlopen

import pytest
from playwright.sync_api import Page, expect


ROOT = Path(__file__).resolve().parents[1]
WEB_URL = "http://127.0.0.1:5173"
API_URL = "http://127.0.0.1:8000/docs"
E2E_ENABLED = os.getenv("QUARRY_RUN_E2E") == "1"

pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(not E2E_ENABLED, reason="Set QUARRY_RUN_E2E=1 to run browser E2E coverage."),
]


def wait_for_url(url: str, *, timeout_s: float = 60.0) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                if 200 <= response.status < 500:
                    return
        except URLError:
            time.sleep(0.5)
    raise RuntimeError(f"Timed out waiting for {url}")


def terminate_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def parse_metric(text: str) -> int:
    match = re.search(r"(\d+)", text)
    if not match:
        raise AssertionError(f"Could not parse numeric metric from: {text}")
    return int(match.group(1))


@pytest.fixture(scope="session")
def backend_server(tmp_path_factory: pytest.TempPathFactory):
    artifact_dir = tmp_path_factory.mktemp("quarry-e2e-artifacts")
    log_dir = ROOT / "data" / "logs" / "playwright"
    log_dir.mkdir(parents=True, exist_ok=True)
    backend_log = (log_dir / "backend.log").open("wb")
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "QUARRY_CORPUS_DIR": str(ROOT / "data" / "corpus"),
            "QUARRY_ARTIFACTS_DIR": str(artifact_dir),
            "QUARRY_RERANK_TOP_K": "3",
            "QUARRY_USE_LOCAL_MODELS": "0",
            "QUARRY_USE_LIVE_LLM": "0",
            "QUARRY_USE_LIVE_EMBEDDINGS": "0",
            "QUARRY_TRACE_LOGS": "0",
        }
    )
    process = subprocess.Popen(
        [str(ROOT / ".venv" / "bin" / "python"), "-m", "uvicorn", "quarry.api.app:app", "--host", "127.0.0.1", "--port", "8000"],
        cwd=ROOT,
        env=env,
        stdout=backend_log,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_url(API_URL)
        yield process
    finally:
        terminate_process(process)
        backend_log.close()


@pytest.fixture(scope="session")
def web_server(backend_server):
    log_dir = ROOT / "data" / "logs" / "playwright"
    log_dir.mkdir(parents=True, exist_ok=True)
    frontend_log = (log_dir / "frontend.log").open("wb")
    process = subprocess.Popen(
        ["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", "5173", "--strictPort"],
        cwd=ROOT / "web",
        stdout=frontend_log,
        stderr=subprocess.STDOUT,
    )
    try:
        wait_for_url(WEB_URL)
        yield process
    finally:
        terminate_process(process)
        frontend_log.close()


def open_diagnostics(page: Page) -> None:
    page.get_by_test_id("open-diagnostics").click()
    expect(page.get_by_test_id("diagnostics-drawer")).to_be_visible()


def close_diagnostics(page: Page) -> None:
    page.get_by_test_id("diagnostics-drawer").get_by_role("button", name="Close").click()
    expect(page.get_by_test_id("diagnostics-drawer")).to_be_hidden()


def run_query(page: Page, query: str | None = None) -> None:
    page.goto(WEB_URL, wait_until="networkidle")
    page.get_by_test_id("query-input").fill(
        query or "How do modular construction and procurement planning affect schedule risk?"
    )
    page.get_by_test_id("run-query").click()
    expect(page.get_by_test_id("conversation-thread")).to_be_visible()
    expect(page.locator("[data-testid^='citation-']").first).to_be_visible(timeout=20000)
    open_diagnostics(page)
    expect(page.get_by_test_id("status-mode")).to_contain_text("response_review")
    close_diagnostics(page)


def open_selection_comment_editor(page: Page) -> None:
    page.evaluate(
        """
        () => {
          const host = document.querySelector('.response-reading-flow');
                    const target = document.querySelector('.response-inline-sentence-copy');
          if (!host || !target) return;

          const walker = document.createTreeWalker(target, NodeFilter.SHOW_TEXT);
          const firstTextNode = walker.nextNode();
          if (!firstTextNode || !firstTextNode.textContent) return;

          const textLength = firstTextNode.textContent.trim().length;
          const endOffset = Math.max(6, Math.min(textLength, 36));
          const range = document.createRange();
          range.setStart(firstTextNode, 0);
          range.setEnd(firstTextNode, endOffset);

          const selection = window.getSelection();
          if (!selection) return;
          selection.removeAllRanges();
          selection.addRange(range);

          host.dispatchEvent(new MouseEvent('mouseup', { bubbles: true }));
        }
        """
    )
    expect(page.get_by_test_id("selection-comment-trigger")).to_be_visible()
    page.get_by_test_id("selection-comment-trigger").click()
    expect(page.get_by_test_id("selection-comment-editor")).to_be_visible()


def test_response_review_dialog_and_feedback_flow(page: Page, web_server) -> None:
    page.set_viewport_size({"width": 1280, "height": 800})
    run_query(page)

    page.locator("[data-testid^='citation-']").first.click()
    expect(page.get_by_test_id("citation-dialog")).to_be_visible()

    page.get_by_test_id("dislike-citation").click()
    page.get_by_test_id("load-alternatives").click()
    assert page.evaluate(
        """
        () => {
          const stack = document.querySelector('.citation-drawer-stack');
          return Boolean(stack && stack.scrollHeight > stack.clientHeight);
        }
        """
    )
    replacement = page.locator("[data-testid^='replace-with-alternative-']").first
    expect(replacement).to_be_visible()
    replacement.click()

    expect(page.get_by_test_id("citation-dialog")).to_be_visible()
    expect(page.locator(".citation-pill.replaced").first).to_be_visible()

    expect(page.get_by_test_id("run-refinement")).to_be_enabled()

    open_selection_comment_editor(page)
    page.get_by_test_id("selection-comment-input").fill(
        "This claim still needs a tighter explanation of how procurement sequencing affects schedule risk."
    )
    page.get_by_test_id("save-selection-comment").click()
    expect(page.get_by_test_id("selection-comment-trigger")).to_have_count(0)
    highlight = page.locator("[data-testid^='annotation-highlight-']").first
    expect(highlight).to_be_visible()
    highlight.click()
    expect(page.get_by_test_id("selection-comment-active-editor")).to_be_visible()
    edit_input = page.locator("[data-testid^='selection-comment-edit-input-']").first
    edit_input.fill("Updated comment for this highlighted passage.")
    page.locator("[data-testid^='update-selection-comment-']").first.click()
    expect(page.get_by_test_id("run-refinement")).to_be_enabled()


def test_unified_refinement_flow(page: Page, web_server) -> None:
    run_query(page, "How do modular construction and procurement planning affect schedule risk?")

    open_diagnostics(page)
    _initial_sentences = parse_metric(page.get_by_test_id("status-sentences").inner_text())
    close_diagnostics(page)

    open_selection_comment_editor(page)
    page.get_by_test_id("selection-comment-input").fill("Please add detail about shutdown planning risks.")
    page.get_by_test_id("save-selection-comment").click()
    page.get_by_test_id("run-refinement").click()
    expect(page.locator(".thread-message.assistant-message")).to_have_count(2)
    expect(page.get_by_text("Please refine the previous answer.")).to_be_visible()

    open_diagnostics(page)
    assert parse_metric(page.get_by_test_id("status-sentences").inner_text()) > 0
    expect(page.get_by_test_id("status-refinements")).to_have_text("1")
    close_diagnostics(page)

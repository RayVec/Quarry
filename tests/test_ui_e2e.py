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
    output_dir = ROOT / "output" / "playwright"
    output_dir.mkdir(parents=True, exist_ok=True)
    backend_log = (output_dir / "backend.log").open("wb")
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
    output_dir = ROOT / "output" / "playwright"
    output_dir.mkdir(parents=True, exist_ok=True)
    frontend_log = (output_dir / "frontend.log").open("wb")
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
    page.get_by_test_id("close-diagnostics").click()
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


def test_response_review_dialog_and_feedback_flow(page: Page, web_server) -> None:
    run_query(page)

    page.locator("[data-testid^='citation-']").first.click()
    expect(page.get_by_test_id("citation-dialog")).to_be_visible()

    page.get_by_test_id("mismatch-note").fill("The first passage is directionally right but I want a stronger local citation.")
    page.get_by_test_id("save-mismatch").click()
    page.get_by_test_id("show-more-citations").click()
    replacement = page.locator("[data-testid^='replace-citation-']").first
    expect(replacement).to_be_visible()
    replacement.click()

    expect(page.get_by_test_id("citation-dialog")).to_be_hidden()
    expect(page.locator(".citation-pill.replaced").first).to_be_visible()

    page.get_by_test_id("toggle-review-panel").click()
    expect(page.get_by_test_id("feedback-summary")).to_contain_text("1 comments captured, 1 citation replacements pending.")

    page.locator("[data-testid^='citation-']").first.click()
    page.get_by_test_id("undo-citation-replacement").click()
    expect(page.locator(".citation-pill.replaced")).to_have_count(0)

    page.locator("[data-testid^='disagree-']").first.click()
    expect(page.get_by_test_id("selection-comment-editor")).to_be_visible()
    page.get_by_test_id("selection-comment-input").fill(
        "This claim still needs a tighter explanation of how procurement sequencing affects schedule risk."
    )
    page.get_by_test_id("save-selection-comment").click()
    expect(page.get_by_test_id("feedback-summary")).to_contain_text("2 comments captured, 0 citation replacements pending.")


def test_unified_refinement_flow(page: Page, web_server) -> None:
    run_query(page, "How do modular construction and procurement planning affect schedule risk?")

    open_diagnostics(page)
    initial_sentences = parse_metric(page.get_by_test_id("status-sentences").inner_text())
    close_diagnostics(page)

    page.get_by_test_id("toggle-review-panel").click()
    page.locator("[data-testid^='disagree-']").first.click()
    expect(page.get_by_test_id("selection-comment-editor")).to_be_visible()
    page.get_by_test_id("selection-comment-input").fill("Please add detail about shutdown planning risks.")
    page.get_by_test_id("save-selection-comment").click()
    expect(page.get_by_test_id("feedback-summary")).to_contain_text("1 comments captured, 0 citation replacements pending.")

    page.get_by_test_id("run-refinement").click()
    expect(page.locator(".thread-message.assistant-message")).to_have_count(2)

    open_diagnostics(page)
    assert parse_metric(page.get_by_test_id("status-sentences").inner_text()) > 0
    expect(page.get_by_test_id("status-refinements")).to_have_text("Refinements: 1")
    close_diagnostics(page)


def test_clarification_suggestions_can_rerun_query(page: Page, web_server) -> None:
    page.goto(WEB_URL, wait_until="networkidle")
    page.get_by_test_id("query-input").fill("schedule")
    page.get_by_test_id("run-query").click()

    expect(page.get_by_test_id("clarification-required")).to_be_visible(timeout=20000)
    suggestion = page.locator("[data-testid^='clarification-suggestion-']").first
    expect(suggestion).to_be_visible()
    suggestion.click()

    expect(page.locator("[data-testid^='citation-']").first).to_be_visible(timeout=20000)
    open_diagnostics(page)
    expect(page.get_by_test_id("status-mode")).to_contain_text("response_review")
    close_diagnostics(page)

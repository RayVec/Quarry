from pathlib import Path

from quarry import logging_utils


def test_resolve_log_dir_uses_test_subdirectory_under_pytest(monkeypatch) -> None:
    monkeypatch.delenv("QUARRY_LOG_DIR", raising=False)
    monkeypatch.delenv("QUARRY_TEST_LOG_DIR", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_logging_utils.py::test (call)")

    resolved = logging_utils.resolve_log_dir(Path("data/logs"), category="runtime")

    assert resolved == Path("data/logs/tests")


def test_resolve_log_dir_prefers_explicit_test_log_dir(monkeypatch) -> None:
    monkeypatch.delenv("QUARRY_LOG_DIR", raising=False)
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_logging_utils.py::test (call)")
    monkeypatch.setenv("QUARRY_TEST_LOG_DIR", "/tmp/quarry-test-logs")

    resolved = logging_utils.resolve_log_dir(Path("data/logs"), category="runtime")

    assert resolved == Path("/tmp/quarry-test-logs")


def test_resolve_log_dir_prefers_explicit_global_log_dir_even_under_pytest(monkeypatch) -> None:
    monkeypatch.setenv("PYTEST_CURRENT_TEST", "tests/test_logging_utils.py::test (call)")
    monkeypatch.setenv("QUARRY_LOG_DIR", "/tmp/quarry-custom-logs")
    monkeypatch.setenv("QUARRY_TEST_LOG_DIR", "/tmp/quarry-test-logs")

    resolved = logging_utils.resolve_log_dir(Path("data/logs"), category="runtime")

    assert resolved == Path("/tmp/quarry-custom-logs")


def test_resolve_log_dir_uses_runtime_and_corpus_subdirectories(monkeypatch) -> None:
    monkeypatch.delenv("QUARRY_LOG_DIR", raising=False)
    monkeypatch.delenv("QUARRY_RUNTIME_LOG_DIR", raising=False)
    monkeypatch.delenv("QUARRY_CORPUS_LOG_DIR", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(logging_utils, "running_under_pytest", lambda: False)

    runtime_dir = logging_utils.resolve_log_dir(Path("data/logs"), category="runtime")
    corpus_dir = logging_utils.resolve_log_dir(Path("data/logs"), category="corpus")

    assert runtime_dir == Path("data/logs/runtime")
    assert corpus_dir == Path("data/logs/corpus")


def test_resolve_log_dir_prefers_category_specific_override(monkeypatch) -> None:
    monkeypatch.delenv("QUARRY_LOG_DIR", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(logging_utils, "running_under_pytest", lambda: False)
    monkeypatch.setenv("QUARRY_RUNTIME_LOG_DIR", "/tmp/quarry-runtime-logs")

    resolved = logging_utils.resolve_log_dir(Path("data/logs"), category="runtime")

    assert resolved == Path("/tmp/quarry-runtime-logs")


def test_configure_logging_tracks_category_specific_files(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("QUARRY_LOG_DIR", raising=False)
    monkeypatch.delenv("QUARRY_TEST_LOG_DIR", raising=False)
    monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
    monkeypatch.setattr(logging_utils, "running_under_pytest", lambda: False)
    monkeypatch.setattr(logging_utils, "_FILE_HANDLER", None)
    monkeypatch.setattr(logging_utils, "_FILE_PATH", None)
    monkeypatch.setattr(logging_utils, "_FILE_HANDLERS", {})
    monkeypatch.setattr(logging_utils, "_FILE_PATHS", {})

    runtime_path = logging_utils.configure_logging(tmp_path, category="runtime")
    corpus_path = logging_utils.configure_logging(tmp_path, category="corpus")

    assert runtime_path == logging_utils.current_log_file("runtime")
    assert corpus_path == logging_utils.current_log_file("corpus")
    assert runtime_path != corpus_path
    assert runtime_path.parent == tmp_path / "runtime"
    assert corpus_path.parent == tmp_path / "corpus"

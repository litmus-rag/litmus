import pytest

pytest_plugins = []


def pytest_configure(config):
    config.addinivalue_line("markers", "live: requires a real LLM call against the configured Azure deployment")


@pytest.fixture(scope="session")
def fixtures_corpus_dir():
    from pathlib import Path

    return str(Path(__file__).parent / "fixtures" / "corpus")


@pytest.fixture(scope="session")
def llm_available():
    from litmus.llm.client import credentials_available

    return credentials_available()


@pytest.fixture(scope="session")
def live_run_dir():
    """Persistent, timestamped output directory for live test artifacts.

    Unlike pytest's built-in `tmp_path` (deleted/rotated automatically and
    buried under a pytest-managed temp root), this directory lives at a
    fixed, predictable location under the repo and is never auto-deleted -
    every live test run's generated eval sets, evaluation results, and LLM
    caches are inspectable afterward. The timestamp in the directory name
    means repeated runs never collide or overwrite each other's output.
    """
    import time
    from pathlib import Path

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(__file__).parent / ".live_runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[live tests] Writing all artifacts for this run to: {run_dir.resolve()}\n")
    return run_dir


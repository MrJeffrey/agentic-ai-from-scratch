"""Static checks: Syntax, imports, and file validation (no API key required)."""

import ast
import importlib.util
import sys
from pathlib import Path

import pytest

TUTORIALS = [
    ("1-interaction-loop", "chat_loop.py"),
    ("2-tool-use", "tool_agent.py"),
    ("3-reasoning-react", "react_agent.py"),
    ("4-memory-context", "memory_agent.py"),
    ("5-rag", "rag_agent.py"),
    ("6-multi-agent", "multi_agent.py"),
    ("7-autonomy-guardrails", "autonomous_agent.py"),
    ("8-evals", "evals.py"),
]

ROOT = Path(__file__).parent.parent


@pytest.mark.parametrize("tutorial_dir,script_name", TUTORIALS)
def test_syntax_valid(tutorial_dir: str, script_name: str):
    """Verify Python syntax is valid (catches SyntaxError)."""
    script_path = ROOT / tutorial_dir / script_name
    assert script_path.exists(), f"Script not found: {script_path}"

    source = script_path.read_text()
    ast.parse(source, filename=str(script_path))


@pytest.mark.parametrize("tutorial_dir,script_name", TUTORIALS)
def test_imports_resolve(tutorial_dir: str, script_name: str, monkeypatch):
    """Verify all imports can be resolved (catches missing dependencies)."""
    script_path = ROOT / tutorial_dir / script_name

    # Mock environment variable to avoid OpenAI client initialization failure
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-for-import-check")

    # Add tutorial directory to path for relative imports
    sys.path.insert(0, str(script_path.parent))

    try:
        spec = importlib.util.spec_from_file_location("module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except ImportError as e:
        pytest.fail(f"Import error in {script_name}: {e}")
    finally:
        sys.path.pop(0)


def test_requirements_files_exist():
    """Verify all tutorials have requirements.txt."""
    for tutorial_dir, _ in TUTORIALS:
        req_file = ROOT / tutorial_dir / "requirements.txt"
        assert req_file.exists(), f"requirements.txt missing in {tutorial_dir}"


def test_start_scripts_exist():
    """Verify all tutorials have start.sh."""
    for tutorial_dir, _ in TUTORIALS:
        start_script = ROOT / tutorial_dir / "start.sh"
        assert start_script.exists(), f"start.sh missing in {tutorial_dir}"

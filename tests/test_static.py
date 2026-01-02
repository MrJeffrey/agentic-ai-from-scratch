"""Static checks: Syntax, imports, and file validation (no API key required)."""

import ast
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent


def discover_tutorials():
    """Dynamically discover tutorials by scanning for numbered directories."""
    tutorials = []
    for path in sorted(ROOT.iterdir()):
        # Match directories like "1-interaction-loop", "2-tool-use", etc.
        if path.is_dir() and re.match(r"^\d+-", path.name):
            # Find the main Python script (not __init__.py)
            py_files = [
                f for f in path.glob("*.py")
                if f.name != "__init__.py" and not f.name.startswith("_")
            ]
            if py_files:
                # Use the first .py file found (typically only one main script)
                tutorials.append((path.name, py_files[0].name))
    return tutorials


TUTORIALS = discover_tutorials()


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

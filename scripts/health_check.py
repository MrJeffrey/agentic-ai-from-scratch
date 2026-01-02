#!/usr/bin/env python3
"""
Local health check script for tutorial workshop.

Usage:
    python scripts/health_check.py [--with-api]

Checks:
    - Python version (3.10+ required)
    - All tutorials have valid syntax
    - All imports can be resolved
    - (Optional) API key works with a single test call
"""

import argparse
import ast
import importlib.util
import os
import re
import sys
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def discover_tutorials(root: Path):
    """Dynamically discover tutorials by scanning for numbered directories."""
    tutorials = []
    for path in sorted(root.iterdir()):
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


def get_root() -> Path:
    """Get the repository root directory."""
    return Path(__file__).parent.parent


def check_python_version() -> bool:
    """Check Python version is 3.10+."""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 10:
        print(f"{GREEN}[PASS]{RESET} Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(
            f"{RED}[FAIL]{RESET} Python 3.10+ required, "
            f"found {version.major}.{version.minor}"
        )
        return False


def check_syntax(root: Path, tutorial_dir: str, script_name: str) -> bool:
    """Check Python file has valid syntax."""
    script_path = root / tutorial_dir / script_name
    if not script_path.exists():
        print(f"{RED}[FAIL]{RESET} File not found: {tutorial_dir}/{script_name}")
        return False

    try:
        ast.parse(script_path.read_text())
        print(f"{GREEN}[PASS]{RESET} Syntax OK: {tutorial_dir}/{script_name}")
        return True
    except SyntaxError as e:
        print(f"{RED}[FAIL]{RESET} Syntax error in {script_name}: {e}")
        return False


def check_imports(root: Path, tutorial_dir: str, script_name: str) -> bool:
    """Check all imports can be resolved."""
    script_path = root / tutorial_dir / script_name

    # Set dummy API key to avoid initialization errors
    os.environ.setdefault("OPENAI_API_KEY", "health-check-dummy-key")

    sys.path.insert(0, str(script_path.parent))
    try:
        spec = importlib.util.spec_from_file_location("module", script_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"{GREEN}[PASS]{RESET} Imports OK: {tutorial_dir}/{script_name}")
        return True
    except ImportError as e:
        print(f"{RED}[FAIL]{RESET} Import error in {script_name}: {e}")
        return False
    except Exception as e:
        # Other errors during module load are OK for health check
        print(
            f"{YELLOW}[WARN]{RESET} Non-import error in {script_name}: "
            f"{type(e).__name__}: {e}"
        )
        return True
    finally:
        sys.path.pop(0)


def check_api_key() -> bool:
    """Check if OPENAI_API_KEY is set."""
    key = os.getenv("OPENAI_API_KEY")
    if key and not key.startswith("health-check"):
        print(f"{GREEN}[PASS]{RESET} OPENAI_API_KEY is set")
        return True
    else:
        print(f"{YELLOW}[SKIP]{RESET} OPENAI_API_KEY not set")
        return False


def check_api_connection() -> bool:
    """Make a minimal API call to verify connection."""
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'ok'"}],
            max_tokens=5,
        )
        if response.choices[0].message.content:
            print(f"{GREEN}[PASS]{RESET} API connection works")
            return True
        else:
            print(f"{RED}[FAIL]{RESET} API returned empty response")
            return False
    except Exception as e:
        print(f"{RED}[FAIL]{RESET} API connection failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Health check for tutorials")
    parser.add_argument(
        "--with-api",
        action="store_true",
        help="Also test API connection (uses ~$0.001)",
    )
    args = parser.parse_args()

    root = get_root()

    print("\n" + "=" * 50)
    print("  Tutorial Workshop Health Check")
    print("=" * 50 + "\n")

    all_passed = True

    # Python version
    all_passed &= check_python_version()
    print()

    # Discover and check each tutorial
    tutorials = discover_tutorials(root)
    if not tutorials:
        print(f"{YELLOW}[WARN]{RESET} No tutorials found in {root}")

    for tutorial_dir, script_name in tutorials:
        passed = check_syntax(root, tutorial_dir, script_name)
        all_passed &= passed
        if passed:
            all_passed &= check_imports(root, tutorial_dir, script_name)

    print()

    # API checks
    has_key = check_api_key()
    if args.with_api and has_key:
        all_passed &= check_api_connection()

    print("\n" + "=" * 50)
    if all_passed:
        print(f"{GREEN}All checks passed!{RESET}")
    else:
        print(f"{RED}Some checks failed. See above for details.{RESET}")
    print("=" * 50 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

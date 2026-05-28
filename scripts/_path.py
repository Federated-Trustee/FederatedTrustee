from __future__ import annotations

import sys
from pathlib import Path


def add_project_root_to_path() -> None:
    """
    Add the project root directory to sys.path.

    This allows scripts inside scripts/ to import modules from src/ when they
    are executed directly, for example:

        python scripts/analyze_model_agreement.py
    """
    project_root = Path(__file__).resolve().parents[1]

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
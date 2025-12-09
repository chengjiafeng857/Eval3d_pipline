from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

from ..config import Settings, get_settings
from ..paths import get_semantic_consistency_dir

console = Console()


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    console.print(f"[blue]Running[/blue]: {' '.join(command)} (cwd={cwd})")
    return subprocess.run(command, cwd=cwd, text=True, check=True, capture_output=True)


def _metric_file(settings: Settings, asset_id: str) -> Path:
    return (
        settings.data_path
        / settings.default_algorithm_name
        / asset_id
        / "semantic_consistency_outputs"
        / "semantic_consistency_metric.txt"
    )


def evaluate_semantic_for_asset(settings: Optional[Settings] = None, asset_id: str = "") -> Path:
    """Run semantic consistency metric for a single asset."""
    settings = settings or get_settings()
    cmd = [
        "python",
        "evaluate.py",
        "--prompt_id",
        asset_id,
        "--base_data_path",
        str(settings.data_path),
        "--algorithm_name",
        settings.default_algorithm_name,
    ]
    try:
        result = _run(cmd, cwd=get_semantic_consistency_dir())
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(result.stderr)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Semantic evaluation failed[/red]: {exc}")
        if exc.stdout:
            console.print(exc.stdout)
        if exc.stderr:
            console.print(exc.stderr)
        raise

    metric_path = _metric_file(settings, asset_id)
    return metric_path if metric_path.exists() else settings.data_path / settings.default_algorithm_name / asset_id


def evaluate_semantic_for_algorithm(settings: Optional[Settings] = None) -> Path:
    """Run semantic consistency for every asset in an algorithm."""
    settings = settings or get_settings()
    cmd = [
        "python",
        "evaluate.py",
        "--base_data_path",
        str(settings.data_path),
        "--algorithm_name",
        settings.default_algorithm_name,
    ]
    try:
        result = _run(cmd, cwd=get_semantic_consistency_dir())
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(result.stderr)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Semantic evaluation failed[/red]: {exc}")
        if exc.stdout:
            console.print(exc.stdout)
        if exc.stderr:
            console.print(exc.stderr)
        raise
    return settings.data_path


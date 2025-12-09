from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from rich.console import Console

from ..config import Settings, get_settings
from ..paths import get_geometric_consistency_dir

console = Console()


def _run(command: list[str], cwd: Path) -> subprocess.CompletedProcess:
    console.print(f"[blue]Running[/blue]: {' '.join(command)} (cwd={cwd})")
    return subprocess.run(command, cwd=cwd, text=True, check=True, capture_output=True)


def run_depth_anything_for_algorithm(settings: Optional[Settings] = None) -> None:
    """Generate depth maps using DepthAnything for all assets of an algorithm."""
    settings = settings or get_settings()
    cmd = [
        "python",
        "run_depth_anything.py",
        "--base_dir",
        str(settings.data_path),
        "--algorithm_name",
        settings.default_algorithm_name,
        "--start_idx",
        "0",
        "--end_idx",
        "100000",
        "--num_gpus",
        str(settings.num_gpus),
        "--available_gpus",
        settings.gpu_ids,
    ]
    try:
        result = _run(cmd, cwd=get_geometric_consistency_dir())
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(result.stderr)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]DepthAnything failed[/red]: {exc}")
        if exc.stdout:
            console.print(exc.stdout)
        if exc.stderr:
            console.print(exc.stderr)
        raise


def evaluate_geometric_for_algorithm(settings: Optional[Settings] = None) -> Path:
    """Run geometric consistency evaluation for every asset in an algorithm."""
    settings = settings or get_settings()
    cmd = [
        "python",
        "evaluate.py",
        "--base_dir",
        str(settings.data_path),
        "--algorithm_name",
        settings.default_algorithm_name,
    ]
    try:
        result = _run(cmd, cwd=get_geometric_consistency_dir())
        if result.stdout:
            console.print(result.stdout)
        if result.stderr:
            console.print(result.stderr)
    except subprocess.CalledProcessError as exc:
        console.print(f"[red]Geometric evaluation failed[/red]: {exc}")
        if exc.stdout:
            console.print(exc.stdout)
        if exc.stderr:
            console.print(exc.stderr)
        raise

    return settings.data_path


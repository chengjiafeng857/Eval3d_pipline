from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console

from .config import Settings, get_settings
from .metrics import aesthetics, geometric, semantic, structural, text3d

console = Console()


def _parse_metric_file(path: Path) -> Optional[float]:
    if not path.exists():
        return None
    content = path.read_text()
    numbers: List[float] = []
    for token in content.replace(",", " ").split():
        try:
            numbers.append(float(token))
        except ValueError:
            continue
    return numbers[-1] if numbers else None


def run_all_metrics_for_asset(
    asset_id: str,
    metrics: Optional[List[str]] = None,
    settings: Optional[Settings] = None,
    geometric_ran: bool = False,
) -> Dict[str, Optional[float]]:
    """Run selected metrics for a single asset and return score mapping."""
    settings = settings or get_settings()
    metrics = metrics or settings.default_metrics

    asset_folder = settings.data_path / settings.default_algorithm_name / asset_id
    results: Dict[str, Optional[float]] = {}

    for metric in metrics:
        if metric == "geometric":
            try:
                if not geometric_ran:
                    geometric.evaluate_geometric_for_algorithm(settings)
                metric_file = asset_folder / "geometric_consistency_metric.txt"
                results["geometric"] = _parse_metric_file(metric_file)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Geometric metric skipped[/yellow]: {exc}")
                results["geometric"] = None
        elif metric == "structural":
            try:
                metric_path = structural.evaluate_structural_for_asset(settings, asset_id)
                results["structural"] = _parse_metric_file(metric_path)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Structural metric skipped[/yellow]: {exc}")
                results["structural"] = None
        elif metric == "semantic":
            try:
                metric_path = semantic.evaluate_semantic_for_asset(settings, asset_id)
                results["semantic"] = _parse_metric_file(metric_path)
            except Exception as exc:  # noqa: BLE001
                console.print(f"[yellow]Semantic metric skipped[/yellow]: {exc}")
                results["semantic"] = None
        elif metric == "aesthetics":
            results["aesthetics"] = aesthetics.compute_aesthetics_for_asset(asset_folder)
        elif metric == "text3d":
            results["text3d"] = text3d.compute_text3d_for_asset(asset_folder, settings=settings)
        else:
            console.print(f"[yellow]Unknown metric '{metric}'[/yellow]")
            results[metric] = None

    scores_path = asset_folder / "eval3d_scores.json"
    scores_path.write_text(json.dumps(results, indent=2))
    return results


def run_all_metrics_for_algorithm(
    metrics: Optional[List[str]] = None,
    settings: Optional[Settings] = None,
) -> Dict[str, Dict[str, Optional[float]]]:
    """Run selected metrics for every asset under the configured algorithm."""
    settings = settings or get_settings()
    metrics = metrics or settings.default_metrics

    algo_dir = settings.data_path / settings.default_algorithm_name
    asset_ids = sorted([p.name for p in algo_dir.iterdir() if p.is_dir()]) if algo_dir.exists() else []

    results: Dict[str, Dict[str, Optional[float]]] = {}
    geometric_ran = False
    if "geometric" in metrics:
        try:
            geometric.evaluate_geometric_for_algorithm(settings)
            geometric_ran = True
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]Geometric evaluation failed for algorithm[/yellow]: {exc}")
            geometric_ran = False

    for asset_id in asset_ids:
        results[asset_id] = run_all_metrics_for_asset(
            asset_id,
            metrics=metrics,
            settings=settings,
            geometric_ran=geometric_ran,
        )

    return results


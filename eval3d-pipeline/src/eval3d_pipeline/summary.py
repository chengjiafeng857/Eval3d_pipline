from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Optional

from rich.console import Console
from rich.table import Table

console = Console()


def _all_metric_keys(results: Dict[str, Dict[str, Optional[float]]]) -> list[str]:
    keys = set()
    for asset_scores in results.values():
        keys.update(asset_scores.keys())
    return sorted(keys)


def write_summary_csv(results: Dict[str, Dict[str, Optional[float]]], path: Path) -> None:
    metrics = _all_metric_keys(results)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["asset_id"] + metrics)
        for asset_id, scores in results.items():
            row = [asset_id] + [scores.get(metric) for metric in metrics]
            writer.writerow(row)


def write_summary_json(results: Dict[str, Dict[str, Optional[float]]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))


def print_summary_table(results: Dict[str, Dict[str, Optional[float]]]) -> None:
    if not results:
        console.print("[yellow]No results to display.[/yellow]")
        return
    metrics = _all_metric_keys(results)
    table = Table(title="Eval3D Metrics")
    table.add_column("asset_id")
    for metric in metrics:
        table.add_column(metric)

    for asset_id, scores in results.items():
        row = [asset_id]
        for metric in metrics:
            value = scores.get(metric)
            row.append(f"{value:.4f}" if isinstance(value, float) else "-")
        table.add_row(*row)

    console.print(table)


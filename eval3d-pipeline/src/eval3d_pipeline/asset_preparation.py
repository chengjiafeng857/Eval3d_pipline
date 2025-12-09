from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from rich.console import Console

console = Console()


@dataclass
class AssetDescriptor:
    asset_id: str
    obx_path: Path
    algorithm_name: str


def discover_obx_assets(root: Path, algorithm_name: str) -> List[AssetDescriptor]:
    """Recursively discover .obx assets under a root directory."""
    assets: List[AssetDescriptor] = []
    for obx_file in root.rglob("*.obx"):
        if not obx_file.is_file():
            continue
        asset_id = obx_file.stem
        assets.append(AssetDescriptor(asset_id=asset_id, obx_path=obx_file.resolve(), algorithm_name=algorithm_name))
        console.print(f"[green]Discovered[/green] asset '{asset_id}' at {obx_file}")
    return assets


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copy2(source, target)


def prepare_obx_for_eval3d(asset: AssetDescriptor, data_path: Path) -> Path:
    """Create an Eval3D-compatible folder for a single asset."""
    asset_folder = data_path / asset.algorithm_name / asset.asset_id
    asset_folder.mkdir(parents=True, exist_ok=True)

    obx_target = asset_folder / "model.obx"
    _link_or_copy(asset.obx_path, obx_target)
    console.print(f"[cyan]Prepared[/cyan] {asset.asset_id} -> {asset_folder}")

    source_dir = asset.obx_path.parent
    video_candidates = [
        source_dir / f"{asset.asset_id}.mp4",
        source_dir / f"{asset.asset_id}_video.mp4",
    ]
    for video in video_candidates:
        if video.exists():
            _link_or_copy(video, asset_folder / "video" / "turntable.mp4")
            break

    question_candidates = [
        source_dir / f"{asset.asset_id}_questions.json",
        source_dir / f"{asset.asset_id}.questions.json",
    ]
    for questions in question_candidates:
        if questions.exists():
            _link_or_copy(questions, asset_folder / "questions" / "questions.json")
            break

    return asset_folder


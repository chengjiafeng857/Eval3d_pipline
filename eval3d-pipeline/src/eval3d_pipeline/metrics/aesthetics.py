from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional, List

import numpy as np
from rich.console import Console

console = Console()

# Lazy imports for heavy dependencies
_IMAGE_REWARD_AVAILABLE = False
_CV2_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    pass


def _extract_frames(video_path: Path, n_frames: int = 60) -> List[Path]:
    """Extract frames from video at regular intervals."""
    if not _CV2_AVAILABLE:
        raise ImportError("opencv-python required for frame extraction")
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    
    temp_dir = Path(tempfile.mkdtemp())
    frame_paths = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = temp_dir / f"{i:04d}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
    
    cap.release()
    return frame_paths


def _compute_image_reward_scores(frame_paths: List[Path], prompt: str = "a 3D render") -> List[float]:
    """Compute ImageReward scores for a list of images."""
    try:
        import ImageReward as RM
    except ImportError:
        raise ImportError("image-reward package required. Install with: pip install image-reward")
    
    console.print("[blue]Loading ImageReward model...[/blue]")
    model = RM.load("ImageReward-v1.0")
    
    image_paths = [str(p) for p in frame_paths]
    
    with __import__('torch').no_grad():
        _, rewards = model.inference_rank(prompt, image_paths)
    
    return rewards


def compute_aesthetics_for_video(video_path: Path, prompt: str = "a 3D render") -> Optional[float]:
    """
    Compute aesthetics score for a video using ImageReward.
    
    This implements the Eval3D aesthetics metric directly without
    relying on the buggy vendor script.
    """
    if not video_path.exists():
        console.print(f"[yellow]Video not found[/yellow]: {video_path}")
        return None
    
    try:
        # Extract frames
        console.print(f"[blue]Extracting frames from video...[/blue]")
        frame_paths = _extract_frames(video_path, n_frames=60)
        
        if not frame_paths:
            console.print("[red]No frames extracted from video[/red]")
            return None
        
        # Compute ImageReward scores
        console.print(f"[blue]Computing ImageReward scores for {len(frame_paths)} frames...[/blue]")
        scores = _compute_image_reward_scores(frame_paths, prompt)
        
        # Pool scores (min over sliding window, then mean) - matches Eval3D method
        scores_array = np.array(scores)
        looped = np.concatenate([scores_array, scores_array, scores_array])
        n = len(scores_array)
        pooled = []
        window = 1
        for i in range(n, 2 * n):
            pooled.append(np.min(looped[i - window:i + window + 1]))
        
        final_score = float(np.mean(pooled))
        
        # Cleanup temp files
        for p in frame_paths:
            p.unlink(missing_ok=True)
        frame_paths[0].parent.rmdir()
        
        console.print(f"[green]Aesthetics score: {final_score:.4f}[/green]")
        return final_score
        
    except ImportError as e:
        console.print(f"[red]Missing dependency[/red]: {e}")
        return None
    except Exception as e:
        console.print(f"[red]Aesthetics evaluation failed[/red]: {e}")
        return None


def compute_aesthetics_for_asset(asset_folder: Path) -> Optional[float]:
    """Locate the turntable video for an asset and compute its aesthetics score."""
    video_path = asset_folder / "video" / "turntable.mp4"
    if not video_path.exists():
        console.print(f"[yellow]Skipping aesthetics (video missing)[/yellow] in {asset_folder}")
        return None
    return compute_aesthetics_for_video(video_path)


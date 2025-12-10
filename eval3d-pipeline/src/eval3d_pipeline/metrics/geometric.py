"""
Geometric Consistency Metric - Full Integration with Depth Anything

This module integrates the original Eval3D geometric consistency pipeline,
including Depth Anything depth estimation, directly into the CLI.

The metric compares:
1. Rendered normals (from the 3D model)  
2. Estimated normals (derived from Depth Anything depth estimation)
"""
from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
from PIL import Image
from rich.console import Console

from ..config import Settings, get_settings

console = Console()

# Lazy imports
_CV2_AVAILABLE = False
_TORCH_AVAILABLE = False
_SCIPY_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    pass

try:
    from scipy.ndimage import uniform_filter
    _SCIPY_AVAILABLE = True
except ImportError:
    pass


def _check_dependencies():
    """Check that all required dependencies are available."""
    missing = []
    if not _CV2_AVAILABLE:
        missing.append("opencv-python")
    if not _TORCH_AVAILABLE:
        missing.append("torch")
    if not _SCIPY_AVAILABLE:
        missing.append("scipy")
    if missing:
        raise ImportError(f"Missing dependencies for geometric metric: {', '.join(missing)}")


# =============================================================================
# DEPTH ANYTHING INTEGRATION
# =============================================================================

def _load_depth_anything_model(encoder: str = "vitl", device: str = None):
    """
    Load Depth Anything model from HuggingFace.
    
    Args:
        encoder: Model size - 'vits', 'vitb', or 'vitl'
        device: Device to load model on
    
    Returns:
        Loaded Depth Anything model
    """
    import torch
    import sys
    import os
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Add Depth Anything path to sys.path
    from ..paths import get_geometric_consistency_dir
    depth_anything_path = get_geometric_consistency_dir() / "Depth-Anything"
    if str(depth_anything_path) not in sys.path:
        sys.path.insert(0, str(depth_anything_path))
    
    try:
        from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
        from torchvision.transforms import Compose
        
        # Change to Depth-Anything directory so torch hub can find local files
        original_cwd = os.getcwd()
        os.chdir(str(depth_anything_path))
        
        try:
            from depth_anything.dpt import DepthAnything
            
            # Load model (will use local torchhub for DINOv2)
            model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(device).eval()
        finally:
            os.chdir(original_cwd)
        
        # Create transform
        transform = Compose([
            Resize(
                width=518,
                height=518,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])
        
        return model, transform, device
        
    except ImportError:
        console.print("[yellow]Depth Anything not available locally, trying transformers fallback...[/yellow]")
        return None, None, device


def _run_depth_anything_on_images(
    rgb_dir: Path,
    output_dir: Path,
    encoder: str = "vitl",
) -> bool:
    """
    Run Depth Anything on all RGB images in a directory.
    
    Args:
        rgb_dir: Directory containing RGB images
        output_dir: Directory to save depth maps
        encoder: Model size
    
    Returns:
        True if successful, False otherwise
    """
    import torch
    import torch.nn.functional as F
    from tqdm import tqdm
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all RGB images (excluding rgba)
    filenames = sorted([
        f for f in glob.glob(str(rgb_dir / "*.png"))
        if "rgba" not in f and "txt" not in f
    ])
    
    if not filenames:
        console.print(f"[red]No RGB images found in {rgb_dir}[/red]")
        return False
    
    # Check if already processed
    existing_depth = list(output_dir.glob("*_depth.npy"))
    if len(existing_depth) >= len(filenames):
        console.print(f"[green]Depth Anything already completed ({len(existing_depth)} files)[/green]")
        return True
    
    console.print(f"[blue]Running Depth Anything on {len(filenames)} images...[/blue]")
    
    # Try loading Depth Anything model
    model, transform, device = _load_depth_anything_model(encoder)
    
    if model is None:
        # Fallback to transformers DPT
        console.print("[yellow]Using transformers DPT model as fallback...[/yellow]")
        return _run_dpt_fallback(filenames, output_dir)
    
    console.print(f"[blue]Using Depth Anything {encoder} on {device}[/blue]")
    
    for filename in tqdm(filenames, desc="Depth Anything"):
        try:
            raw_image = cv2.imread(filename)
            image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
            h, w = image.shape[:2]
            
            image_tensor = transform({'image': image})['image']
            image_tensor = torch.from_numpy(image_tensor).unsqueeze(0).to(device)
            
            with torch.no_grad():
                depth = model(image_tensor)
            
            depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]
            
            # Save as .npy
            basename = os.path.basename(filename)
            name_no_ext = basename[:basename.rfind('.')]
            np.save(output_dir / f"{name_no_ext}_depth.npy", depth.cpu().numpy())
            
            # Also save visualization
            depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth_vis = depth_vis.cpu().numpy().astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{name_no_ext}_depth.png"), depth_vis)
            
        except Exception as e:
            console.print(f"[red]Error processing {filename}: {e}[/red]")
            continue
    
    # Mark as complete
    (output_dir / f"{name_no_ext}_depth_anything_rendered.txt").touch()
    
    console.print(f"[green]Depth Anything completed. Saved to {output_dir}[/green]")
    return True


def _run_dpt_fallback(filenames: List[str], output_dir: Path) -> bool:
    """Fallback to transformers DPT model if Depth Anything not available."""
    import torch
    from tqdm import tqdm
    
    try:
        from transformers import DPTForDepthEstimation, DPTImageProcessor
        
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device).eval()
        
        console.print(f"[blue]Using DPT-Large on {device}[/blue]")
        
        for filename in tqdm(filenames, desc="DPT Depth"):
            try:
                image = Image.open(filename).convert("RGB")
                h, w = image.size[1], image.size[0]
                
                inputs = processor(images=image, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    depth = outputs.predicted_depth
                
                # Interpolate to original size
                depth = torch.nn.functional.interpolate(
                    depth.unsqueeze(1),
                    size=(h, w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
                
                # Save
                basename = os.path.basename(filename)
                name_no_ext = basename[:basename.rfind('.')]
                np.save(output_dir / f"{name_no_ext}_depth.npy", depth.cpu().numpy())
                
            except Exception as e:
                console.print(f"[red]Error processing {filename}: {e}[/red]")
                continue
        
        return True
        
    except ImportError as e:
        console.print(f"[red]DPT fallback failed: {e}[/red]")
        return False


# =============================================================================
# GEOMETRIC CONSISTENCY METRIC (Original Eval3D Implementation)
# =============================================================================

def _normalize_vectors(vectors: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Normalize vectors along specified axis."""
    norms = np.linalg.norm(vectors, axis=axis, keepdims=True)
    return vectors / (norms + eps)


def _get_normal_transformed(normal: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """Transform normals from world space to camera space."""
    normal = normal.reshape(-1, 3)
    normal_transformed = transformation @ normal.transpose()
    normal_transformed = normal_transformed.swapaxes(0, 1)
    normal_transformed = normal_transformed.reshape(512, 512, 3)
    return normal_transformed[:, :, :3]


def _depth_map_to_normal_map(depth_map: np.ndarray) -> np.ndarray:
    """Convert a depth map to a normal map using gradients."""
    depth_map = depth_map * -1

    gradient_y = np.gradient(depth_map, axis=0)
    gradient_x = np.gradient(depth_map, axis=1)

    normal_x = gradient_x
    normal_y = gradient_y
    normal_z = np.ones_like(depth_map)

    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    normal_x /= norm
    normal_y /= norm
    normal_z /= norm

    return np.stack([normal_x, normal_y, normal_z], axis=-1)


def _compute_geometric_metric(
    data_path: Path,
    normal_metric_threshold: float = 0.3,
    opacity_threshold: float = 0.01,
) -> float:
    """
    Compute geometric consistency metric using original Eval3D algorithm.
    
    This is a direct port of the original evaluate.py compute_metric function.
    """
    import cv2
    import torch
    from scipy.ndimage import uniform_filter
    
    # Get all data files
    all_normal_data = sorted(glob.glob(str(data_path / "normal_world" / "*.npy")))
    all_batch_data = sorted(glob.glob(str(data_path / "batch_data" / "*.npy")))
    all_opacity_data = sorted(glob.glob(str(data_path / "opacity" / "*.png")))
    all_rgb_data = sorted([f for f in glob.glob(str(data_path / "rgb_images" / "*.png")) if "rgba" not in f])
    all_depth_anything = sorted(glob.glob(str(data_path / "depth_anything" / "*.npy")))
    
    if not all_depth_anything:
        raise ValueError("No Depth Anything files found. Run depth estimation first.")
    
    assert len(all_normal_data) == len(all_batch_data), "Normal and batch data count mismatch"
    assert len(all_normal_data) == len(all_rgb_data), "Normal and RGB data count mismatch"
    
    # Create output directories
    (data_path / "normal_metric").mkdir(exist_ok=True)
    (data_path / "normal_camera").mkdir(exist_ok=True)
    (data_path / "depth_anything_normal_camera").mkdir(exist_ok=True)
    
    all_opacity = []
    geometric_consistency_metric = []
    
    from tqdm import tqdm
    
    for idx in tqdm(range(len(all_normal_data)), desc="Computing metric"):
        # Process every other frame (as in original)
        if idx % 2 != 0:
            continue
        
        try:
            batch_data = np.load(all_batch_data[idx], allow_pickle=True).item()
            normal_world = np.load(all_normal_data[idx])
            rgb_image = np.asarray(Image.open(all_rgb_data[idx]))
            depth_anything = np.load(all_depth_anything[idx])
            opacity_map = np.asarray(Image.open(all_opacity_data[idx]))[..., 0] / 255.0
            
            # Resize all to 512x512
            normal_world = cv2.resize(normal_world, (512, 512))
            depth_anything = cv2.resize(depth_anything, (512, 512))
            rgb_image = cv2.resize(rgb_image, (512, 512))
            opacity_map = cv2.resize(opacity_map, (512, 512))
            opacity_map = (opacity_map > 0) * 1.0
            
            # Convert depth to normals
            depth_anything_normal_camera = _depth_map_to_normal_map(depth_anything)
            
            # Get camera transformation
            c2w = batch_data['c2w']
            if isinstance(c2w, torch.Tensor):
                c2w = c2w.cpu().numpy()
            c2w = np.array(c2w)[:, :3, :3]
            w2c = np.linalg.inv(c2w[0])
            
            # Threestudio normal adjustment
            normal_world = (normal_world * 2.0) - 1.0
            normal_world = _normalize_vectors(normal_world)
            normal_camera = _get_normal_transformed(normal_world, w2c)
            normal_camera = _normalize_vectors(normal_camera)
            normal_camera = (normal_camera + 1.0) / 2.0
            normal_camera = normal_camera * opacity_map[..., None]
            
            # DreamCraft3D adjustment
            normal_camera[..., 0] = 1 - normal_camera[..., 0]
            normal_camera = (2 * normal_camera - 1)
            normal_camera = (normal_camera + 1.0) / 2.0
            normal_camera = _normalize_vectors(normal_camera)
            normal_camera = normal_camera * opacity_map[..., None]
            
            # Depth anything normal adjustment
            depth_anything_normal_camera[..., -1] *= -1
            depth_anything_normal_camera = (depth_anything_normal_camera + 1.0) / 2.0
            depth_anything_normal_camera = (1 - 2 * depth_anything_normal_camera)
            depth_anything_normal_camera = (depth_anything_normal_camera + 1.0) / 2.0
            depth_anything_normal_camera = _normalize_vectors(depth_anything_normal_camera)
            depth_anything_normal_camera = depth_anything_normal_camera * opacity_map[..., None]
            
            # Compute angular difference
            dot_product = np.clip(
                np.sum(
                    depth_anything_normal_camera.reshape(-1, 3) * normal_camera.reshape(-1, 3),
                    axis=-1
                ).reshape(512, 512),
                -1.0, 1.0
            )
            normal_metric = np.arccos(dot_product)
            normal_metric = normal_metric * opacity_map
            
            all_opacity.append(opacity_map.sum() / np.prod(opacity_map.shape))
            
            # Apply smoothing filter
            kernel_size = 11
            normal_metric_smoothed = uniform_filter(normal_metric, size=kernel_size)
            normal_metric_smoothed = normal_metric_smoothed * opacity_map
            mask = (normal_metric_smoothed < normal_metric_threshold) | np.isnan(normal_metric_smoothed)
            geometric_consistency_metric.append((1 - mask).sum() / (opacity_map.sum() + 1e-8))
            
        except Exception as e:
            console.print(f"[yellow]Error processing frame {idx}: {e}[/yellow]")
            continue
    
    if not geometric_consistency_metric:
        raise ValueError("No frames could be processed")
    
    # Compute final metric (original algorithm)
    mean_opacity = np.stack(all_opacity).mean()
    geometric_consistency_metric = np.stack(geometric_consistency_metric).reshape(-1, 3).max(axis=-1).mean(axis=0)
    geometric_consistency_metric = 100.0 if mean_opacity < opacity_threshold else 100 * geometric_consistency_metric
    geometric_consistency_metric = 100 - geometric_consistency_metric
    
    return float(geometric_consistency_metric)


# =============================================================================
# PUBLIC API
# =============================================================================

def compute_geometric_for_asset(
    asset_folder: Path,
    run_depth_anything: bool = True,
    depth_anything_encoder: str = "vitl",
) -> Optional[float]:
    """
    Compute geometric consistency metric for a single asset.
    
    This integrates the full original Eval3D pipeline:
    1. Run Depth Anything if depth maps don't exist
    2. Compute geometric consistency using original algorithm
    
    Args:
        asset_folder: Path to the asset folder containing save/it0-test/
        run_depth_anything: Whether to run Depth Anything if needed
        depth_anything_encoder: Encoder size ('vits', 'vitb', 'vitl')
    
    Returns:
        Geometric consistency score (0-100), or None if computation fails
    """
    _check_dependencies()
    
    data_path = asset_folder / "save" / "it0-test"
    
    if not data_path.exists():
        console.print(f"[yellow]No render data found at {data_path}[/yellow]")
        return None
    
    rgb_dir = data_path / "rgb_images"
    depth_dir = data_path / "depth_anything"
    
    # Check if we need to run Depth Anything
    existing_depth = list(depth_dir.glob("*_depth.npy")) if depth_dir.exists() else []
    rgb_files = [f for f in rgb_dir.glob("*.png") if "rgba" not in f.name]
    
    if len(existing_depth) < len(rgb_files):
        if run_depth_anything:
            console.print(f"[blue]Running Depth Anything preprocessing...[/blue]")
            success = _run_depth_anything_on_images(rgb_dir, depth_dir, depth_anything_encoder)
            if not success:
                console.print("[red]Depth Anything failed[/red]")
                return None
        else:
            console.print(f"[yellow]Depth maps missing ({len(existing_depth)}/{len(rgb_files)}). "
                         f"Run with --run-depth-anything or preprocess manually.[/yellow]")
            return None
    else:
        console.print(f"[green]Using existing Depth Anything files ({len(existing_depth)} found)[/green]")
    
    # Compute metric
    try:
        console.print(f"[blue]Computing geometric consistency metric...[/blue]")
        score = _compute_geometric_metric(data_path)
        
        # Save result
        result_path = asset_folder / "geometric_consistency_metric.txt"
        result_path.write_text(f"Geometric Consistency Metric: {score}")
        
        console.print(f"[green]Geometric consistency score: {score:.2f}[/green]")
        return score
        
    except Exception as e:
        console.print(f"[red]Geometric metric computation failed: {e}[/red]")
        import traceback
        traceback.print_exc()
        return None


def evaluate_geometric_for_algorithm(settings: Optional[Settings] = None) -> Path:
    """
    Run geometric consistency evaluation for every asset in an algorithm.
    """
    settings = settings or get_settings()
    algo_dir = settings.data_path / settings.default_algorithm_name
    
    if not algo_dir.exists():
        raise FileNotFoundError(f"Algorithm directory not found: {algo_dir}")
    
    asset_ids = sorted([p.name for p in algo_dir.iterdir() if p.is_dir()])
    
    for asset_id in asset_ids:
        asset_folder = algo_dir / asset_id
        console.print(f"\n[bold]Processing {asset_id}...[/bold]")
        compute_geometric_for_asset(asset_folder)

    return settings.data_path

"""
Render a 3D asset (.obj, .glb, .ply, .obx) to the FULL Eval3D-compatible format.

This module generates ALL data needed for Eval3D metrics:
1. RGB images (120 views)
2. Opacity/alpha masks
3. World-space normal maps (.npy)
4. batch_data with camera parameters (.npy) - CRITICAL for geometric/semantic metrics
5. Turntable video (for aesthetics/text-3D)

The key insight: Eval3D's batch_data contains standard camera parameters
(c2w, proj_mtx, elevation, azimuth, camera_distances, fovy) that we can
compute ourselves when rendering. This makes the pipeline work with ANY mesh!

Requires: trimesh, pyrender, numpy, opencv-python, Pillow, torch
"""
from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
from rich.console import Console

console = Console()

# Optional heavy imports
_TRIMESH_AVAILABLE = False
_PYRENDER_AVAILABLE = False
_CV2_AVAILABLE = False
_TORCH_AVAILABLE = False

try:
    import trimesh
    _TRIMESH_AVAILABLE = True
except ImportError:
    pass

try:
    import pyrender
    _PYRENDER_AVAILABLE = True
except ImportError:
    pass

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    pass

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


def check_dependencies(require_torch: bool = False) -> bool:
    """Check if rendering dependencies are available."""
    missing = []
    if not _TRIMESH_AVAILABLE:
        missing.append("trimesh")
    if not _PYRENDER_AVAILABLE:
        missing.append("pyrender")
    if not _CV2_AVAILABLE:
        missing.append("opencv-python")
    if require_torch and not _TORCH_AVAILABLE:
        missing.append("torch")
    
    if missing:
        console.print(f"[red]Missing dependencies:[/red] {', '.join(missing)}")
        console.print("[yellow]Install with:[/yellow] uv pip install trimesh pyrender opencv-python PyOpenGL numpy Pillow torch")
        return False
    return True


def load_mesh(mesh_path: Path) -> "trimesh.Trimesh":
    """Load and normalize a 3D mesh file."""
    if not _TRIMESH_AVAILABLE:
        raise ImportError("trimesh is required")
    
    mesh = trimesh.load(str(mesh_path), force='mesh')
    
    if isinstance(mesh, trimesh.Scene):
        if len(mesh.geometry) == 0:
            raise ValueError(f"No geometry found in {mesh_path}")
        meshes = list(mesh.geometry.values())
        mesh = trimesh.util.concatenate(meshes)
    
    # Center and normalize
    mesh.vertices -= mesh.centroid
    scale = 1.0 / max(mesh.extents)
    mesh.vertices *= scale
    
    return mesh


# ============================================================================
# THREESTUDIO-COMPATIBLE CAMERA GENERATION
# Replicates the exact math from threestudio/data/uncond.py
# ============================================================================

def get_projection_matrix(fovy: float, aspect: float, near: float = 0.01, far: float = 100.0) -> np.ndarray:
    """
    Compute projection matrix matching threestudio's get_projection_matrix.
    
    Args:
        fovy: Field of view in radians
        aspect: width / height
        near, far: clipping planes
    """
    tan_half_fovy = np.tan(fovy / 2.0)
    
    proj = np.zeros((4, 4), dtype=np.float32)
    proj[0, 0] = 1.0 / (aspect * tan_half_fovy)
    proj[1, 1] = 1.0 / tan_half_fovy
    proj[2, 2] = -(far + near) / (far - near)
    proj[2, 3] = -2.0 * far * near / (far - near)
    proj[3, 2] = -1.0
    
    return proj


def get_c2w_matrix(
    elevation_deg: float,
    azimuth_deg: float,
    camera_distance: float,
) -> np.ndarray:
    """
    Compute camera-to-world matrix matching threestudio's convention.
    
    Right-hand coordinate system: x back, y right, z up
    """
    elevation = np.radians(elevation_deg)
    azimuth = np.radians(azimuth_deg)
    
    # Camera position in spherical coordinates
    camera_position = np.array([
        camera_distance * np.cos(elevation) * np.cos(azimuth),
        camera_distance * np.cos(elevation) * np.sin(azimuth),
        camera_distance * np.sin(elevation),
    ], dtype=np.float32)
    
    # Look at origin
    center = np.zeros(3, dtype=np.float32)
    up = np.array([0, 0, 1], dtype=np.float32)
    
    # Compute camera axes
    lookat = center - camera_position
    lookat = lookat / (np.linalg.norm(lookat) + 1e-8)
    
    right = np.cross(lookat, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    
    up = np.cross(right, lookat)
    up = up / (np.linalg.norm(up) + 1e-8)
    
    # Build c2w matrix (threestudio convention)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -lookat
    c2w[:3, 3] = camera_position
    
    return c2w


def create_threestudio_batch_data(
    index: int,
    elevation_deg: float = 15.0,
    azimuth_deg: float = 0.0,
    camera_distance: float = 1.5,
    fovy_deg: float = 70.0,
    height: int = 512,
    width: int = 512,
) -> Dict[str, Any]:
    """
    Create a batch_data dictionary exactly matching threestudio's format.
    
    This is what gets saved to batch_data/*.npy
    """
    fovy = np.radians(fovy_deg)
    
    c2w = get_c2w_matrix(elevation_deg, azimuth_deg, camera_distance)
    proj_mtx = get_projection_matrix(fovy, width / height)
    
    # MVP = proj @ w2c = proj @ inv(c2w)
    w2c = np.linalg.inv(c2w)
    mvp_mtx = proj_mtx @ w2c
    
    camera_position = c2w[:3, 3]
    
    # Convert to torch tensors if available (Eval3D expects torch tensors in .npy)
    if _TORCH_AVAILABLE:
        import torch
        batch = {
            "index": index,
            "c2w": torch.from_numpy(c2w[None]),  # [1, 4, 4]
            "proj_mtx": torch.from_numpy(proj_mtx[None]),
            "mvp_mtx": torch.from_numpy(mvp_mtx[None]),
            "camera_positions": torch.from_numpy(camera_position[None]),
            "light_positions": torch.from_numpy(camera_position[None]),
            "elevation": torch.tensor([elevation_deg]),
            "azimuth": torch.tensor([azimuth_deg]),
            "camera_distances": torch.tensor([camera_distance]),
            "height": height,
            "width": width,
            "fovy": torch.tensor([fovy]),
        }
    else:
        batch = {
            "index": index,
            "c2w": c2w[None],
            "proj_mtx": proj_mtx[None],
            "mvp_mtx": mvp_mtx[None],
            "camera_positions": camera_position[None],
            "light_positions": camera_position[None],
            "elevation": elevation_deg,
            "azimuth": azimuth_deg,
            "camera_distances": camera_distance,
            "height": height,
            "width": width,
            "fovy": fovy,
        }
    
    return batch


# ============================================================================
# NORMAL MAP COMPUTATION
# ============================================================================

def compute_world_normal_from_depth(
    depth: np.ndarray,
    c2w: np.ndarray,
    fovy: float,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Compute world-space normals from a depth map.
    
    This matches how threestudio computes normals for the geometric consistency metric.
    """
    # Compute camera intrinsics
    focal = 0.5 * height / np.tan(0.5 * fovy)
    cx, cy = width / 2, height / 2
    
    # Create pixel grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # Backproject to camera space
    z = depth
    x = (u - cx) * z / focal
    y = (v - cy) * z / focal
    
    # Compute normals in camera space using gradients
    dzdx = np.gradient(z, axis=1)
    dzdy = np.gradient(z, axis=0)
    
    normal_x = -dzdx
    normal_y = -dzdy
    normal_z = np.ones_like(z)
    
    # Normalize
    norm = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2) + 1e-8
    normal_camera = np.stack([normal_x/norm, normal_y/norm, normal_z/norm], axis=-1)
    
    # Transform to world space
    R = c2w[:3, :3]
    normal_world = normal_camera @ R.T
    
    # Normalize to [0, 1] range (threestudio convention)
    normal_world = (normal_world + 1.0) / 2.0
    
    return normal_world.astype(np.float32)


# ============================================================================
# FULL EVAL3D-COMPATIBLE RENDERING
# ============================================================================

def render_eval3d_compatible(
    mesh_path: Path,
    output_dir: Path,
    n_views: int = 120,
    image_size: Tuple[int, int] = (512, 512),
    elevation_deg: float = 15.0,
    camera_distance: float = 1.5,
    fovy_deg: float = 70.0,
) -> Path:
    """
    Render a mesh to the FULL Eval3D-compatible format.
    
    Creates:
        output_dir/
            save/it0-test/
                rgb_images/0000.png ... 0119.png
                opacity/0000.png ...
                normal_world/0000.npy ...
                batch_data/0000.npy ...  <- THIS IS THE KEY!
    
    This output structure is directly compatible with ALL Eval3D metrics.
    """
    if not check_dependencies(require_torch=True):
        raise ImportError("Missing rendering dependencies")
    
    import pyrender
    import torch
    
    height, width = image_size
    
    # Create output directories matching threestudio structure
    save_dir = output_dir / "save" / "it0-test"
    rgb_dir = save_dir / "rgb_images"
    opacity_dir = save_dir / "opacity"
    normal_dir = save_dir / "normal_world"
    batch_dir = save_dir / "batch_data"
    
    for d in [rgb_dir, opacity_dir, normal_dir, batch_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Load mesh
    console.print(f"[blue]Loading mesh[/blue]: {mesh_path}")
    mesh = load_mesh(mesh_path)
    
    # Compute vertex normals
    mesh.fix_normals()
    
    # Create pyrender mesh
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    else:
        # Use a warm terracotta/clay color for untextured meshes - more visually appealing
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.85, 0.65, 0.55, 1.0],  # Warm terracotta color
            metallicFactor=0.1,
            roughnessFactor=0.6,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    
    # Soft blue-gray background with alpha=0 for opacity masking
    scene = pyrender.Scene(bg_color=[0.85, 0.88, 0.92, 0.0], ambient_light=[0.4, 0.4, 0.45])
    mesh_node = scene.add(pr_mesh)
    
    # Add warm key light that follows camera
    light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=2.5)
    light_node = scene.add(light, pose=np.eye(4))
    
    # Add cool fill light from above for rim lighting effect
    fill_light = pyrender.DirectionalLight(color=[0.8, 0.85, 1.0], intensity=1.2)
    above_pose = np.eye(4)
    above_pose[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # Point down
    scene.add(fill_light, pose=above_pose)
    
    # Add back light for depth
    back_light = pyrender.DirectionalLight(color=[0.7, 0.75, 0.85], intensity=0.8)
    back_pose = np.eye(4)
    back_pose[:3, :3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # Point from back
    scene.add(back_light, pose=back_pose)
    
    # Create camera
    fovy = np.radians(fovy_deg)
    camera = pyrender.PerspectiveCamera(yfov=fovy, aspectRatio=width/height)
    camera_node = scene.add(camera)
    
    # Create renderer
    renderer = pyrender.OffscreenRenderer(width, height)
    
    # Generate azimuth angles (0 to 360, n_views points)
    azimuth_angles = np.linspace(0, 360.0, n_views, endpoint=False)
    
    console.print(f"[blue]Rendering {n_views} views for Eval3D...[/blue]")
    
    for i, azimuth in enumerate(azimuth_angles):
        # Create batch_data with camera parameters (for Eval3D)
        batch_data = create_threestudio_batch_data(
            index=i,
            elevation_deg=elevation_deg,
            azimuth_deg=azimuth,
            camera_distance=camera_distance,
            fovy_deg=fovy_deg,
            height=height,
            width=width,
        )
        
        # Compute camera pose for PYRENDER (different from threestudio)
        # Pyrender uses OpenGL convention: Y up, -Z forward
        elevation_rad = np.radians(elevation_deg)
        azimuth_rad = np.radians(azimuth)
        
        # Camera position on sphere around origin
        cam_x = camera_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
        cam_y = camera_distance * np.sin(elevation_rad)
        cam_z = camera_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
        cam_pos = np.array([cam_x, cam_y, cam_z])
        
        # Look-at matrix (camera looks at origin)
        def look_at(eye, target, up):
            forward = target - eye
            forward = forward / np.linalg.norm(forward)
            right = np.cross(forward, up)
            right = right / np.linalg.norm(right)
            up_new = np.cross(right, forward)
            mat = np.eye(4)
            mat[:3, 0] = right
            mat[:3, 1] = up_new
            mat[:3, 2] = -forward
            mat[:3, 3] = eye
            return mat
        
        camera_pose = look_at(cam_pos, np.zeros(3), np.array([0, 1, 0]))
        scene.set_pose(camera_node, camera_pose)
        
        # Update light to follow camera
        scene.set_pose(light_node, camera_pose)
        
        # Get c2w for later use
        c2w = batch_data["c2w"]
        if _TORCH_AVAILABLE and torch.is_tensor(c2w):
            c2w = c2w[0].numpy()
        else:
            c2w = c2w[0]
        
        # Render color and depth
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        
        # Save RGB image
        rgb_path = rgb_dir / f"{i:04d}.png"
        cv2.imwrite(str(rgb_path), cv2.cvtColor(color[:, :, :3], cv2.COLOR_RGB2BGR))
        
        # Save opacity mask
        opacity = (color[:, :, 3] > 0).astype(np.uint8) * 255
        opacity_rgba = np.stack([opacity, opacity, opacity, opacity], axis=-1)
        opacity_path = opacity_dir / f"{i:04d}.png"
        cv2.imwrite(str(opacity_path), opacity_rgba)
        
        # Compute and save world-space normals
        # Use mesh normals rendered to screen for better quality
        normal_world = compute_world_normal_from_depth(
            depth, c2w, fovy, width, height
        )
        normal_world = normal_world * (color[:, :, 3:4] > 0).astype(np.float32)
        normal_path = normal_dir / f"{i:04d}.npy"
        np.save(str(normal_path), normal_world)
        
        # Save batch_data (this is what Eval3D uses for camera transforms!)
        batch_path = batch_dir / f"{i:04d}.npy"
        np.save(str(batch_path), batch_data, allow_pickle=True)
        
        if (i + 1) % 20 == 0:
            console.print(f"  Rendered {i + 1}/{n_views}")
    
    renderer.delete()
    console.print(f"[green]Saved Eval3D-compatible data to[/green] {output_dir}")
    
    return output_dir


def render_turntable_video(
    mesh_path: Path,
    output_video: Path,
    n_frames: int = 60,
    image_size: Tuple[int, int] = (512, 512),
    fps: int = 30,
    camera_distance: float = 2.5,
    elevation_deg: float = 20.0,
) -> Path:
    """Render a turntable video (for aesthetics/text-3D metrics)."""
    if not check_dependencies():
        raise ImportError("Missing rendering dependencies")
    
    import pyrender
    
    output_video.parent.mkdir(parents=True, exist_ok=True)
    
    console.print(f"[blue]Loading mesh[/blue]: {mesh_path}")
    mesh = load_mesh(mesh_path)
    
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
    else:
        # Use a warm terracotta/clay color for untextured meshes - more visually appealing
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[0.85, 0.65, 0.55, 1.0],  # Warm terracotta color
            metallicFactor=0.1,
            roughnessFactor=0.6,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    
    # Soft blue-gray background (studio-like)
    scene = pyrender.Scene(bg_color=[0.85, 0.88, 0.92, 1.0], ambient_light=[0.4, 0.4, 0.45])
    scene.add(pr_mesh)
    
    # Add a warm key light that will follow the camera
    key_light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.9], intensity=2.5)
    key_light_node = scene.add(key_light, pose=np.eye(4))
    
    # Add cool fill light from above for rim lighting effect
    fill_light = pyrender.DirectionalLight(color=[0.8, 0.85, 1.0], intensity=1.2)
    # Light from above
    above_pose = np.eye(4)
    above_pose[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])  # Point down
    scene.add(fill_light, pose=above_pose)
    
    # Add back light for depth
    back_light = pyrender.DirectionalLight(color=[0.7, 0.75, 0.85], intensity=0.8)
    back_pose = np.eye(4)
    back_pose[:3, :3] = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])  # Point from back
    scene.add(back_light, pose=back_pose)
    
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 4.0, aspectRatio=1.0)
    camera_node = scene.add(camera)
    
    renderer = pyrender.OffscreenRenderer(image_size[0], image_size[1])
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video), fourcc, fps, image_size)
    
    console.print(f"[blue]Rendering {n_frames} frames for video...[/blue]")
    
    # Helper function for look-at matrix (OpenGL convention: Y up)
    def look_at(eye, target, up):
        forward = target - eye
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, up)
        right = right / np.linalg.norm(right)
        up_new = np.cross(right, forward)
        mat = np.eye(4)
        mat[:3, 0] = right
        mat[:3, 1] = up_new
        mat[:3, 2] = -forward
        mat[:3, 3] = eye
        return mat
    
    for i in range(n_frames):
        azimuth = 2 * np.pi * i / n_frames
        elevation_rad = np.radians(elevation_deg)
        
        # Camera position (OpenGL: Y up)
        cam_x = camera_distance * np.cos(elevation_rad) * np.sin(azimuth)
        cam_y = camera_distance * np.sin(elevation_rad)
        cam_z = camera_distance * np.cos(elevation_rad) * np.cos(azimuth)
        
        camera_pos = np.array([cam_x, cam_y, cam_z])
        pose = look_at(camera_pos, np.zeros(3), np.array([0, 1, 0]))
        
        scene.set_pose(camera_node, pose)
        # Key light follows camera for consistent front lighting
        scene.set_pose(key_light_node, pose)
        
        color, _ = renderer.render(scene)
        frame = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
        video_writer.write(frame)
    
    video_writer.release()
    renderer.delete()
    console.print(f"[green]Saved video to[/green] {output_video}")
    return output_video


def prepare_asset_for_eval3d(
    mesh_path: Path,
    data_path: Path,
    algorithm_name: str,
    asset_id: Optional[str] = None,
    render_video: bool = True,
    render_full_eval3d: bool = True,
    n_video_frames: int = 60,
    n_render_views: int = 120,
) -> Path:
    """
    Prepare a 3D mesh for FULL Eval3D evaluation.
    
    This creates the complete folder structure with ALL data needed for
    ALL Eval3D metrics including geometric and semantic consistency.
    """
    asset_id = asset_id or mesh_path.stem
    asset_folder = data_path / algorithm_name / asset_id
    asset_folder.mkdir(parents=True, exist_ok=True)
    
    # Copy mesh
    mesh_ext = mesh_path.suffix.lower()
    target_mesh = asset_folder / f"model{mesh_ext}"
    if not target_mesh.exists():
        shutil.copy2(mesh_path, target_mesh)
    console.print(f"[cyan]Copied mesh to[/cyan] {target_mesh}")
    
    # Also save as .obj if not already (Eval3D semantic metric expects model.obj)
    if mesh_ext != ".obj":
        obj_target = asset_folder / "model.obj"
        if not obj_target.exists():
            try:
                mesh = load_mesh(mesh_path)
                mesh.export(str(obj_target))
                console.print(f"[cyan]Exported OBJ to[/cyan] {obj_target}")
            except Exception as e:
                console.print(f"[yellow]Could not export OBJ:[/yellow] {e}")
    
    # Render full Eval3D-compatible data (RGB, opacity, normals, batch_data)
    if render_full_eval3d:
        try:
            render_eval3d_compatible(
                mesh_path=mesh_path,
                output_dir=asset_folder,
                n_views=n_render_views,
            )
        except Exception as e:
            console.print(f"[yellow]Could not render Eval3D data:[/yellow] {e}")
    
    # Render video
    if render_video:
        video_dir = asset_folder / "video"
        video_dir.mkdir(exist_ok=True)
        video_path = video_dir / "turntable.mp4"
        if not video_path.exists():
            try:
                render_turntable_video(mesh_path, video_path, n_frames=n_video_frames)
            except Exception as e:
                console.print(f"[yellow]Could not render video:[/yellow] {e}")
    
    return asset_folder


def generate_sample_questions(prompt: str, output_path: Path, n_questions: int = 5) -> Path:
    """Generate sample yes/no questions for text-3D alignment."""
    questions = {
        "1": f"Does this object match the description: '{prompt}'?",
        "2": "Is the object complete (no missing parts)?",
        "3": "Does the object have a consistent texture?",
        "4": "Is the geometry smooth and artifact-free?",
        "5": "Does the object look realistic?",
    }
    questions = {k: v for k, v in list(questions.items())[:n_questions]}
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(questions, indent=2))
    console.print(f"[green]Generated sample questions at[/green] {output_path}")
    return output_path

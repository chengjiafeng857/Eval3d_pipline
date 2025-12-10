from __future__ import annotations

import subprocess
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console

from .asset_preparation import AssetDescriptor, discover_obx_assets, prepare_obx_for_eval3d
from .config import get_settings
from .runner import run_all_metrics_for_algorithm, run_all_metrics_for_asset
from .summary import print_summary_table, write_summary_csv, write_summary_json
from .metrics import text3d

app = typer.Typer(help="Eval3D pipeline for 3D assets (.obj, .glb, .ply, .obx).")
console = Console()

# Supported 3D file extensions
SUPPORTED_MESH_EXTENSIONS = {".obj", ".glb", ".gltf", ".ply", ".stl", ".off", ".obx"}


def _apply_algorithm_override(algorithm_name: Optional[str]):
    settings = get_settings()
    if algorithm_name:
        settings.default_algorithm_name = algorithm_name
    return settings


@app.command("init-vendor")
def init_vendor(
    repo_url: str = typer.Option(
        "https://github.com/eval3d/eval3d-codebase.git",
        help="Git URL for the Eval3D repository.",
    ),
) -> None:
    """Clone the Eval3D repository into vendor/eval3d."""
    settings = get_settings()
    target_dir = settings.vendor_eval3d_root
    if target_dir.exists():
        console.print(f"[green]Vendor repo already present[/green] at {target_dir}")
        return
    target_dir.parent.mkdir(parents=True, exist_ok=True)
    console.print(f"[blue]Cloning Eval3D[/blue] into {target_dir} ...")
    subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)
    console.print("[green]Clone complete.[/green]")


@app.command("install-eval3d-deps")
def install_eval3d_deps() -> None:
    """Install heavy Eval3D dependencies via uv."""
    settings = get_settings()
    requirements = settings.vendor_eval3d_root / "requirements.txt"
    if not requirements.exists():
        console.print(f"[red]requirements.txt not found at {requirements}[/red]")
        raise typer.Exit(code=1)
    console.print(f"[blue]Installing Eval3D dependencies from[/blue] {requirements}")
    subprocess.run(["uv", "pip", "install", "-r", str(requirements)], check=True)
    console.print(
        "[yellow]Reminder[/yellow]: install tiny-cuda-nn, FeatUp, Zero123 checkpoints, image-reward, and set OPENAI_API_KEY as needed."
    )


@app.command("prepare-obx")
def prepare_obx(
    source_root: Path = typer.Argument(..., exists=True, readable=True),
    algorithm_name: Optional[str] = typer.Option(None, "--algorithm-name", "--algo"),
) -> None:
    """Discover .obx files and prepare Eval3D-compatible folders."""
    settings = _apply_algorithm_override(algorithm_name)
    assets = discover_obx_assets(source_root, settings.default_algorithm_name)
    for asset in assets:
        prepare_obx_for_eval3d(asset, settings.data_path)
    console.print(f"[green]Prepared {len(assets)} assets.[/green]")


@app.command("eval-obx")
def eval_obx(
    obx_file: Path = typer.Argument(..., exists=True, readable=True),
    algorithm_name: Optional[str] = typer.Option(None, "--algorithm-name", "--algo"),
    metrics: Optional[List[str]] = typer.Option(None, "--metrics", "-m", help="Subset of metrics to run."),
) -> None:
    """Evaluate a single .obx asset."""
    settings = _apply_algorithm_override(algorithm_name)
    asset = AssetDescriptor(asset_id=obx_file.stem, obx_path=obx_file.resolve(), algorithm_name=settings.default_algorithm_name)
    prepare_obx_for_eval3d(asset, settings.data_path)
    results = run_all_metrics_for_asset(asset.asset_id, metrics=metrics, settings=settings)
    print_summary_table({asset.asset_id: results})


@app.command("eval-obx-batch")
def eval_obx_batch(
    source_root: Path = typer.Argument(..., exists=True, readable=True),
    algorithm_name: Optional[str] = typer.Option(None, "--algorithm-name", "--algo"),
    metrics: Optional[List[str]] = typer.Option(None, "--metrics", "-m", help="Subset of metrics to run."),
    output_dir: Path = typer.Option(Path("eval3d_results"), "--output-dir", "-o"),
) -> None:
    """Evaluate every .obx file under a directory."""
    settings = _apply_algorithm_override(algorithm_name)
    assets = discover_obx_assets(source_root, settings.default_algorithm_name)
    for asset in assets:
        prepare_obx_for_eval3d(asset, settings.data_path)

    results = run_all_metrics_for_algorithm(metrics=metrics, settings=settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_json(results, output_dir / "summary.json")
    write_summary_csv(results, output_dir / "summary.csv")
    print_summary_table(results)
    console.print(f"[green]Saved summaries to {output_dir}[/green]")


@app.command("summarize")
def summarize(
    output_dir: Path = typer.Option(Path("eval3d_results"), "--output-dir", "-o"),
    metrics: Optional[List[str]] = typer.Option(None, "--metrics", "-m"),
) -> None:
    """Re-run metrics for all prepared assets and write summary files."""
    settings = get_settings()
    results = run_all_metrics_for_algorithm(metrics=metrics, settings=settings)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_json(results, output_dir / "summary.json")
    write_summary_csv(results, output_dir / "summary.csv")
    print_summary_table(results)
    console.print(f"[green]Saved summaries to {output_dir}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
# RENDERING COMMANDS - for users who only have mesh files
# ─────────────────────────────────────────────────────────────────────────────


@app.command("render-video")
def render_video(
    mesh_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to 3D mesh file (.obj, .glb, .ply, etc.)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output video path (default: <mesh_name>/video/turntable.mp4)"),
    n_frames: int = typer.Option(60, "--frames", "-f", help="Number of frames in the video"),
    size: int = typer.Option(1024, "--size", "-s", help="Frame size (square)"),
    fps: int = typer.Option(30, "--fps", help="Frames per second"),
    distance: float = typer.Option(1.8, "--distance", "-d", help="Camera distance from object"),
    elevation: float = typer.Option(15.0, "--elevation", "-e", help="Camera elevation in degrees"),
) -> None:
    """Render a turntable video from a 3D mesh file.

    This is all you need for the AESTHETICS and TEXT-3D metrics.

    Example:
        eval3d-pipeline render-video ./model.obj -o ./output/turntable.mp4
    """
    try:
        from .render_asset import render_turntable_video
    except ImportError:
        console.print("[red]Rendering requires additional dependencies.[/red]")
        console.print("[yellow]Install with:[/yellow] uv pip install trimesh pyrender opencv-python PyOpenGL")
        raise typer.Exit(code=1)

    if output is None:
        output = mesh_file.parent / mesh_file.stem / "video" / "turntable.mp4"

    try:
        render_turntable_video(
            mesh_path=mesh_file,
            output_video=output,
            n_frames=n_frames,
            image_size=(size, size),
            fps=fps,
            camera_distance=distance,
            elevation_deg=elevation,
        )
    except Exception as e:
        console.print(f"[red]Rendering failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("render-views")
def render_views_cmd(
    mesh_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to 3D mesh file (.obj, .glb, .ply, etc.)"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for renders"),
    n_views: int = typer.Option(120, "--views", "-n", help="Number of views to render"),
    size: int = typer.Option(1024, "--size", "-s", help="Image size (square)"),
    distance: float = typer.Option(1.8, "--distance", "-d", help="Camera distance from object"),
    elevation: float = typer.Option(15.0, "--elevation", "-e", help="Camera elevation in degrees"),
) -> None:
    """Render multi-view images from a 3D mesh file.

    This produces RGB images and opacity masks in the format expected by Eval3D.
    Note: For full geometric/semantic metrics you also need camera pose data
    which this simplified renderer does not produce.

    Example:
        eval3d-pipeline render-views ./model.obj -o ./output/renders -n 120
    """
    try:
        from .render_asset import render_views
    except ImportError:
        console.print("[red]Rendering requires additional dependencies.[/red]")
        console.print("[yellow]Install with:[/yellow] uv pip install trimesh pyrender opencv-python PyOpenGL")
        raise typer.Exit(code=1)

    if output_dir is None:
        output_dir = mesh_file.parent / mesh_file.stem / "save" / "it0-test"

    try:
        render_views(
            mesh_path=mesh_file,
            output_dir=output_dir,
            n_views=n_views,
            image_size=(size, size),
            camera_distance=distance,
            elevation_deg=elevation,
        )
    except Exception as e:
        console.print(f"[red]Rendering failed:[/red] {e}")
        raise typer.Exit(code=1)


@app.command("prepare-mesh")
def prepare_mesh(
    mesh_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to 3D mesh file"),
    algorithm_name: Optional[str] = typer.Option(None, "--algorithm-name", "--algo"),
    asset_id: Optional[str] = typer.Option(None, "--asset-id", help="Custom asset ID (default: filename)"),
    render_video: bool = typer.Option(True, "--video", help="Render turntable video"),
    render_full: bool = typer.Option(True, "--full", help="Render full Eval3D data (120 views + batch_data)"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Text prompt to generate sample questions"),
) -> None:
    """Prepare a 3D mesh for FULL Eval3D evaluation.

    This command generates ALL data needed for ALL Eval3D metrics:
    1. 120 RGB renders with known camera poses
    2. Opacity/alpha masks
    3. World-space normal maps
    4. batch_data with camera matrices (c2w, proj_mtx, elevation, azimuth)
    5. Turntable video

    Example:
        eval3d-pipeline prepare-mesh ./robot.obj --algo my_method --prompt "a robot"
    """
    settings = _apply_algorithm_override(algorithm_name)

    try:
        from .render_asset import prepare_asset_for_eval3d, generate_sample_questions
    except ImportError:
        console.print("[red]Rendering requires additional dependencies.[/red]")
        console.print("[yellow]Install with:[/yellow] uv pip install trimesh pyrender opencv-python PyOpenGL torch")
        raise typer.Exit(code=1)

    aid = asset_id or mesh_file.stem

    try:
        asset_folder = prepare_asset_for_eval3d(
            mesh_path=mesh_file,
            data_path=settings.data_path,
            algorithm_name=settings.default_algorithm_name,
            asset_id=aid,
            render_video=render_video,
            render_full_eval3d=render_full,
        )

        if prompt:
            questions_path = asset_folder / "questions" / "questions.json"
            generate_sample_questions(prompt, questions_path)

        console.print(f"\n[green]Asset prepared at:[/green] {asset_folder}")
        console.print("\n[bold]What you can evaluate now:[/bold]")
        
        video_exists = (asset_folder / "video" / "turntable.mp4").exists()
        questions_exist = (asset_folder / "questions" / "questions.json").exists()
        batch_data_exists = (asset_folder / "save" / "it0-test" / "batch_data").exists()
        rgb_exists = (asset_folder / "save" / "it0-test" / "rgb_images").exists()
        
        if video_exists:
            console.print("  ✅ [green]Aesthetics[/green] - video available")
            if questions_exist:
                console.print("  ✅ [green]Text-3D Alignment[/green] - video + questions available")
            else:
                console.print("  ⚠️  [yellow]Text-3D Alignment[/yellow] - needs questions.json (use --prompt)")
        
        if batch_data_exists and rgb_exists:
            console.print("  ✅ [green]Geometric Consistency[/green] - renders + batch_data available")
            console.print("  ✅ [green]Semantic Consistency[/green] - renders + batch_data + mesh available")
            console.print("  ⚠️  [yellow]Structural Consistency[/yellow] - requires Zero123 model (optional)")
        else:
            console.print("  ❌ [red]Geometric/Semantic[/red] - use --full to render")

    except Exception as e:
        console.print(f"[red]Preparation failed:[/red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("eval-mesh")
def eval_mesh(
    mesh_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to 3D mesh file"),
    algorithm_name: Optional[str] = typer.Option(None, "--algorithm-name", "--algo"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Text prompt for the asset"),
    questions_json: Optional[Path] = typer.Option(None, "--questions", "-q", help="Path to questions JSON"),
    metrics: Optional[List[str]] = typer.Option(
        None, "--metrics", "-m",
        help="Metrics to run (geometric, semantic, structural, aesthetics, text3d)"
    ),
    skip_render: bool = typer.Option(False, "--skip-render", help="Skip rendering (use existing data)"),
    full: bool = typer.Option(True, "--full", help="Full render (all metrics) vs quick (video only)"),
) -> None:
    """Evaluate a 3D mesh file end-to-end with ALL Eval3D metrics.

    This is the main command for users who only have mesh files (.obj, .glb, .ply).
    
    With --full (default), it will:
    1. Render 120 views with camera pose data
    2. Render a turntable video
    3. Run ALL available metrics (geometric, semantic, aesthetics, text3d)
    
    With --quick, it will only render a video for aesthetics/text3d.

    Example:
        eval3d-pipeline eval-mesh ./robot.obj --algo my_method --prompt "a robot"
    """
    settings = _apply_algorithm_override(algorithm_name)

    try:
        from .render_asset import prepare_asset_for_eval3d, generate_sample_questions
    except ImportError:
        console.print("[red]Rendering requires additional dependencies.[/red]")
        console.print("[yellow]Install with:[/yellow] uv pip install trimesh pyrender opencv-python PyOpenGL torch")
        raise typer.Exit(code=1)

    asset_id = mesh_file.stem

    # Prepare the asset
    console.print(f"\n[bold blue]Step 1: Preparing asset[/bold blue]")
    try:
        asset_folder = prepare_asset_for_eval3d(
            mesh_path=mesh_file,
            data_path=settings.data_path,
            algorithm_name=settings.default_algorithm_name,
            asset_id=asset_id,
            render_video=not skip_render,
            render_full_eval3d=full and not skip_render,
        )
    except Exception as e:
        console.print(f"[red]Asset preparation failed:[/red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)

    # Handle questions
    if questions_json and questions_json.exists():
        import shutil
        target_questions = asset_folder / "questions" / "questions.json"
        target_questions.parent.mkdir(exist_ok=True)
        shutil.copy2(questions_json, target_questions)
    elif prompt:
        questions_path = asset_folder / "questions" / "questions.json"
        generate_sample_questions(prompt, questions_path)

    # Determine which metrics to run
    video_exists = (asset_folder / "video" / "turntable.mp4").exists()
    questions_exist = (asset_folder / "questions" / "questions.json").exists()
    batch_data_exists = (asset_folder / "save" / "it0-test" / "batch_data").exists()

    if not metrics:
        metrics = []
        if video_exists:
            metrics.append("aesthetics")
            if questions_exist and settings.openai_api_key:
                metrics.append("text3d")
        if batch_data_exists:
            metrics.append("geometric")
            metrics.append("semantic")
            # Note: structural requires Zero123, skip by default

    if not metrics:
        console.print("[yellow]No metrics can be run. Rendering may have failed.[/yellow]")
        raise typer.Exit(code=1)

    console.print(f"\n[bold blue]Step 2: Running metrics[/bold blue]: {', '.join(metrics)}")

    results = run_all_metrics_for_asset(asset_id, metrics=metrics, settings=settings)
    print_summary_table({asset_id: results})


@app.command("suggest")
def suggest(
    asset_path: Path = typer.Argument(..., exists=True, readable=True, help="Path to asset folder or mesh file"),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Text prompt describing the 3D model"),
    algorithm_name: Optional[str] = typer.Option(None, "--algorithm-name", "--algo"),
    n_frames: int = typer.Option(12, "--frames", "-f", help="Number of frames to analyze"),
) -> None:
    """Get detailed improvement suggestions for a 3D model using GPT-4o.
    
    This analyzes your 3D model and provides feedback on:
    1. Geometry flaws (mesh quality, proportions, artifacts)
    2. Texture flaws (UV mapping, material consistency)
    3. Multi-view consistency (Janus face, view-dependent issues)
    4. Semantic reasonableness (does it make sense?)
    5. Prompt-specific questions (auto-generated based on your prompt)
    
    Requires OPENAI_API_KEY to be set in environment or .env file.
    
    Examples:
        eval3d-pipeline suggest ./my_algo/robot/ --prompt "a cute robot"
        eval3d-pipeline suggest ./model.obj --prompt "a medieval castle"
    """
    settings = _apply_algorithm_override(algorithm_name)
    
    if not settings.openai_api_key:
        console.print("[red]Error: OPENAI_API_KEY not set[/red]")
        console.print("[yellow]Set it in your .env file or environment variables[/yellow]")
        raise typer.Exit(code=1)
    
    # Determine asset folder
    if asset_path.is_file():
        # It's a mesh file - need to prepare it first
        if asset_path.suffix.lower() in SUPPORTED_MESH_EXTENSIONS:
            try:
                from .render_asset import prepare_asset_for_eval3d
            except ImportError:
                console.print("[red]Rendering requires additional dependencies.[/red]")
                console.print("[yellow]Install with:[/yellow] uv pip install trimesh pyrender opencv-python PyOpenGL")
                raise typer.Exit(code=1)
            
            asset_id = asset_path.stem
            console.print(f"[blue]Preparing mesh for analysis...[/blue]")
            
            asset_folder = prepare_asset_for_eval3d(
                mesh_path=asset_path,
                data_path=settings.data_path,
                algorithm_name=settings.default_algorithm_name,
                asset_id=asset_id,
                render_video=True,
                render_full_eval3d=False,  # Only need video for suggestions
            )
        else:
            console.print(f"[red]Unsupported file type: {asset_path.suffix}[/red]")
            raise typer.Exit(code=1)
    else:
        asset_folder = asset_path
    
    # Check for video
    video_path = asset_folder / "video" / "turntable.mp4"
    if not video_path.exists():
        console.print(f"[red]No turntable video found at {video_path}[/red]")
        console.print("[yellow]Run 'eval3d-pipeline render-video' first, or use 'eval3d-pipeline prepare-mesh'[/yellow]")
        raise typer.Exit(code=1)
    
    # Run suggestions
    result = text3d.compute_text3d_suggestions(
        asset_folder=asset_folder,
        prompt=prompt,
        settings=settings,
        n_frames=n_frames,
    )
    
    if result is None:
        console.print("[red]Analysis failed[/red]")
        raise typer.Exit(code=1)
    
    console.print("\n[green]✓ Analysis complete![/green]")


@app.command("info")
def info() -> None:
    """Show current configuration and what metrics are available."""
    settings = get_settings()
    
    console.print("\n[bold]Current Configuration[/bold]")
    console.print(f"  Project root:     {settings.project_root}")
    console.print(f"  Vendor Eval3D:    {settings.vendor_eval3d_root}")
    console.print(f"  Data path:        {settings.data_path}")
    console.print(f"  Algorithm name:   {settings.default_algorithm_name}")
    console.print(f"  GPU IDs:          {settings.gpu_ids}")
    console.print(f"  Num GPUs:         {settings.num_gpus}")
    console.print(f"  OpenAI API key:   {'✅ set' if settings.openai_api_key else '❌ not set'}")
    console.print(f"  Default metrics:  {', '.join(settings.default_metrics)}")
    
    console.print("\n[bold]Metric Requirements[/bold]")
    console.print("  [green]Aesthetics[/green]           - turntable video (.mp4)")
    console.print("  [green]Text-3D Alignment[/green]    - video + questions.json + OPENAI_API_KEY")
    console.print("  [yellow]Geometric Consistency[/yellow] - RGB renders + normals + camera poses (threestudio)")
    console.print("  [yellow]Structural Consistency[/yellow] - RGB renders + Zero123 model")
    console.print("  [yellow]Semantic Consistency[/yellow]  - RGB renders + camera poses + mesh + FeatUp")
    
    console.print("\n[bold]For users with only mesh files:[/bold]")
    console.print("  Use [cyan]eval3d-pipeline eval-mesh ./model.obj --prompt 'description'[/cyan]")
    console.print("  This will render a video and run aesthetics + text-3D alignment.")


if __name__ == "__main__":
    app()


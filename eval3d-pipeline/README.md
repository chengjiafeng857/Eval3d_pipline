# eval3d-pipeline

CLI wrapper around [Eval3D](https://github.com/eval3d/eval3d-codebase) that lets you evaluate **ANY 3D mesh file** with ALL Eval3D metrics.

> **".mesh in, scores out"** - No threestudio required!

## 🎯 What This Solves

The official Eval3D was designed for threestudio-generated assets with specific internal outputs. **This pipeline generates all required data from just a mesh file**, making Eval3D work with assets from:
- Online AI 3D generators (Meshy, CSM, Tripo, etc.)
- Any image-to-3D tool
- Any mesh file (.obj, .glb, .ply, .stl, .obx)

## ✅ What You Can Evaluate

| Metric | Status | What It Measures |
|--------|--------|------------------|
| **Aesthetics** | ✅ Works | Visual quality (ImageReward on video frames) |
| **Text-3D Alignment** | ✅ Works | Does it match the text prompt? (GPT-4o VQA) |
| **Geometric Consistency** | ✅ Works | Texture-geometry alignment (rendered vs depth normals) |
| **Semantic Consistency** | ✅ Works | DINO feature consistency across views |
| **Structural Consistency** | ⚠️ Optional | Requires Zero123 model |

---

## Quick Start

### 1. Install
```bash
cd eval3d-pipeline

# Install core dependencies
uv sync

# Install rendering dependencies
uv pip install trimesh pyrender opencv-python PyOpenGL numpy Pillow torch

# For text-3D alignment
export OPENAI_API_KEY=your_key_here
```

### 2. Evaluate a Mesh (One Command!)
```bash
# This renders 120 views, generates batch_data, and runs metrics
uv run eval3d-pipeline eval-mesh ./robot.obj --algo my_method --prompt "a robot"
```

### What Happens Under the Hood
1. **Renders 120 views** of your mesh at known camera poses
2. **Generates batch_data** with camera matrices (c2w, proj_mtx, elevation, azimuth)
3. **Computes normal maps** in world space
4. **Renders a turntable video**
5. **Runs all available metrics**

---

## Commands

### Main Command: `eval-mesh`
```bash
# Full evaluation (all metrics)
uv run eval3d-pipeline eval-mesh ./model.obj --algo my_method --prompt "description"

# Quick evaluation (video-based metrics only)
uv run eval3d-pipeline eval-mesh ./model.obj --algo my_method --quick

# Specific metrics only
uv run eval3d-pipeline eval-mesh ./model.obj -m geometric -m semantic
```

### Prepare Without Evaluating
```bash
# Generate all Eval3D-compatible data
uv run eval3d-pipeline prepare-mesh ./model.obj --algo my_method --full

# This creates:
# <data_path>/my_method/model/
#   save/it0-test/
#     rgb_images/0000.png ... 0119.png
#     opacity/0000.png ...
#     normal_world/0000.npy ...
#     batch_data/0000.npy ...  <- Camera parameters!
#   video/turntable.mp4
#   model.obj
```

### Render Only
```bash
# Just render a turntable video
uv run eval3d-pipeline render-video ./model.obj -o ./output.mp4

# Render multi-view images
uv run eval3d-pipeline render-views ./model.obj -o ./renders/
```

### Check Configuration
```bash
uv run eval3d-pipeline info
```

---

## Configuration

Create a `.env` file:
```env
EVAL3D_DATA_PATH=/path/to/evaluation/data
EVAL3D_ALGO_NAME=my_algo
EVAL3D_GPU_IDS=0
EVAL3D_NUM_GPUS=1
OPENAI_API_KEY=your_openai_key
```

---

## How It Works

### The Key Insight

Eval3D's metrics require `batch_data/*.npy` files containing camera parameters:
```python
{
    "c2w": camera_to_world_matrix,      # 4x4 matrix
    "proj_mtx": projection_matrix,      # 4x4 matrix
    "elevation": 15.0,                  # degrees
    "azimuth": 0.0 to 360.0,           # degrees
    "camera_distances": 1.5,
    "fovy": 70.0,                       # degrees
    ...
}
```

These are **standard camera parameters** that we can compute ourselves when rendering! This pipeline:

1. Uses the same camera math as threestudio (`uncond.py`)
2. Renders views with PyRender using identical parameters
3. Saves batch_data in the exact format Eval3D expects

### Metric Details

**Geometric Consistency**: Compares surface normals from 3D geometry vs normals estimated from depth (DepthAnything). Measures texture-geometry alignment.

**Semantic Consistency**: Projects DINO features onto mesh vertices across views, measures feature variance. High variance = semantic inconsistency (e.g., Janus problem).

**Aesthetics**: Extracts frames from turntable video, scores with ImageReward.

**Text-3D Alignment**: GPT-4o answers yes/no questions about the video frames.

---

## Using with Eval3D Scripts Directly

If you want to use the generated data with the original Eval3D scripts:

```bash
# 1. Prepare your mesh
uv run eval3d-pipeline prepare-mesh ./model.obj --algo my_algo --full

# 2. Clone Eval3D (if not already)
uv run eval3d-pipeline init-vendor

# 3. Run Eval3D scripts directly
cd vendor/eval3d/Eval3D/geometric_consistency
python evaluate.py --base_dir $EVAL3D_DATA_PATH --algorithm_name my_algo
```

---

## Example Output

```
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━━┳━━━━━━━━┓
┃ asset_id     ┃ geometric  ┃ semantic   ┃ aesthetics  ┃ text3d ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━━╇━━━━━━━━┩
│ robot        │ 85.23      │ 78.45      │ 0.7523      │ -      │
└──────────────┴────────────┴────────────┴─────────────┴────────┘
```

---

## Troubleshooting

### "Missing rendering dependencies"
```bash
uv pip install trimesh pyrender opencv-python PyOpenGL numpy Pillow torch
```

### "Could not create GL context" (headless server)
```bash
apt-get install libosmesa6-dev
export PYOPENGL_PLATFORM=osmesa
```

### Geometric/Semantic metrics fail
Make sure you ran with `--full` flag to generate batch_data. Check that:
- `<asset>/save/it0-test/batch_data/` exists
- `<asset>/save/it0-test/rgb_images/` has 120 images

### Zero123 / Structural Consistency
Structural consistency requires running Zero123 to generate novel views. This requires:
1. Download [Stable Zero123 checkpoint](https://huggingface.co/stabilityai/stable-zero123)
2. Install Zero123 dependencies
3. This metric is optional - the others work without it

---

## Citation

If you use this tool, please cite Eval3D:

```bibtex
@article{cvpr2025eval3d,
    title={Eval3D: Interpretable and Fine-grained Evaluation for 3D Generation},
    author={Duggal, Shivam and Hu, Yushi and Michel, Oscar and others},
    journal={CVPR},
    year={2025},
}
```

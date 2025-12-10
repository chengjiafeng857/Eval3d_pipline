# Eval3D Pipeline - Code Structure & Implementation Details

This document provides a comprehensive overview of the codebase structure and implementation details for future code generation and maintenance.

---

## Project Structure

```
eval3d-pipeline/
├── pyproject.toml              # Project dependencies and metadata
├── README.md                   # User documentation
├── CODE_STRUCTURE.md           # This file - technical documentation
├── data/                       # Evaluation data folder
│   └── <algorithm>/
│       └── <asset_id>/
│           ├── model.obj       # 3D mesh file
│           ├── model.glb       # Alternative format
│           ├── material.mtl    # Material definitions
│           ├── material_0.png  # Texture images
│           ├── video/
│           │   └── turntable.mp4
│           ├── save/
│           │   └── it0-test/
│           │       ├── rgb_images/     # 120 rendered views
│           │       ├── opacity/        # Alpha masks
│           │       ├── normal_world/   # World-space normals
│           │       └── batch_data/     # Camera parameters
│           ├── questions/
│           │   └── questions.json      # VQA questions
│           ├── vqa_frames/             # Extracted video frames for VQA
│           ├── suggestion_frames/      # Frames for suggestions analysis
│           ├── text3d_results.json     # VQA metric results
│           ├── text3d_suggestions.json # Detailed AI suggestions
│           └── eval3d_scores.json      # All metric scores
└── src/
    └── eval3d_pipeline/
        ├── __init__.py
        ├── cli.py              # Typer CLI commands
        ├── config.py           # Pydantic settings
        ├── paths.py            # Path utilities
        ├── render_asset.py     # 3D rendering with PyRender
        ├── runner.py           # Metric orchestration
        ├── summary.py          # Result aggregation
        └── metrics/
            ├── __init__.py
            ├── aesthetics.py   # ImageReward scoring
            ├── geometric.py    # Geometric consistency
            ├── semantic.py     # Semantic consistency (DINO)
            ├── structural.py   # Structural consistency (Zero123)
            └── text3d.py       # Text-3D alignment & suggestions
```

---

## Core Modules

### 1. `cli.py` - Command Line Interface

**Framework**: Typer

**Key Commands**:

| Command | Function | Description |
|---------|----------|-------------|
| `eval-mesh` | `eval_mesh()` | End-to-end mesh evaluation |
| `suggest` | `suggest_improvements_cmd()` | GPT-5.1 quality suggestions |
| `render-video` | `render_video()` | Turntable video rendering |
| `render-views` | `render_views_cmd()` | Multi-view image rendering |
| `prepare-mesh` | `prepare_mesh_cmd()` | Prepare asset without evaluation |
| `info` | `info()` | Show configuration |

**Default Settings** (as of Dec 2024):
```python
# Video rendering
size: int = 1024           # Resolution (was 512)
distance: float = 1.8      # Camera distance (was 2.5)
elevation: float = 15.0    # Camera elevation (was 20.0)
n_frames: int = 60         # Video frames
fps: int = 30              # Frames per second

# Multi-view rendering
n_views: int = 120         # Number of views for Eval3D
```

---

### 2. `config.py` - Configuration Management

**Framework**: Pydantic Settings

```python
class Settings(BaseSettings):
    # Paths
    eval3d_data_path: Path = Path("./data")
    eval3d_algo_name: str = "my_algo"
    
    # Compute
    eval3d_gpu_ids: str = "0"
    eval3d_num_gpus: int = 1
    
    # API Keys
    openai_api_key: str = ""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
```

**Environment Variables**:
- `EVAL3D_DATA_PATH` - Base path for evaluation data
- `EVAL3D_ALGO_NAME` - Algorithm/method name
- `OPENAI_API_KEY` - Required for text3d and suggestions

---

### 3. `render_asset.py` - 3D Rendering

**Dependencies**: trimesh, pyrender, opencv-python, numpy

**Key Functions**:

#### `load_mesh(mesh_path: Path) -> trimesh.Trimesh`
- Loads OBJ, GLB, PLY, STL files
- Handles Scene objects (multiple geometries)
- Centers and normalizes mesh to unit scale

#### `render_turntable_video(...)`
```python
def render_turntable_video(
    mesh_path: Path,
    output_video: Path,
    n_frames: int = 60,
    image_size: Tuple[int, int] = (1024, 1024),  # High resolution
    fps: int = 30,
    camera_distance: float = 1.8,   # Closer for better framing
    elevation_deg: float = 15.0,    # Lower angle
) -> Path:
```

**Visual Detection** (texture vs clay material):
```python
# Check for various visual types
has_visual = False
if hasattr(mesh, 'visual') and mesh.visual is not None:
    visual_kind = getattr(mesh.visual, 'kind', None)
    if visual_kind == 'texture':     # Has texture maps
        has_visual = True
    elif visual_kind == 'vertex':    # Has vertex colors
        has_visual = True
    elif visual_kind == 'face':      # Has face colors
        has_visual = True

if has_visual:
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)
else:
    # Default clay material for untextured meshes
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.85, 0.65, 0.55, 1.0],  # Warm terracotta
        metallicFactor=0.1,
        roughnessFactor=0.6,
    )
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
```

#### `render_eval3d_compatible(...)`
Renders 120 views with full Eval3D-compatible data:
- RGB images
- Opacity masks
- World-space normal maps
- Camera batch_data (c2w, proj_mtx, etc.)

---

### 4. `metrics/text3d.py` - Text-3D Alignment & Suggestions

**Dependencies**: openai, cv2, PIL, base64

**Model Constants**:
```python
GPT_MODEL = "gpt-5.1"           # Primary model for VQA and suggestions
GPT_MODEL_MINI = "gpt-5-mini"   # Not currently used
GPT_MODEL_FAST = "gpt-5-nano"   # For high-throughput tasks
```

#### Core Functions

##### `_extract_key_frames(...)`
```python
def _extract_key_frames(
    video_path: Path, 
    n_frames: int = 12,
    output_dir: Optional[Path] = None,  # Save frames for inspection
    prefix: str = "frame"
) -> List[Path]:
```
- Extracts evenly-spaced frames from video
- Saves to `output_dir` if provided (for debugging)
- Returns list of frame paths

##### `_encode_image_to_base64(image_path: Path) -> str`
- Converts image to base64 for OpenAI API
- Uses JPEG encoding at 85% quality

##### `_call_gpt4o_vqa(...)`
```python
def _call_gpt4o_vqa(
    image_path: Path,
    questions: Dict[str, str],
    api_key: str
) -> Dict[int, str]:
```

**API Call Pattern**:
```python
response = client.chat.completions.create(
    model=GPT_MODEL,  # gpt-5.1
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {...}},
            {"type": "text", "text": prompt}
        ]
    }],
    max_completion_tokens=1500,  # NOT max_tokens for GPT-5
    # NOTE: temperature not supported with GPT-5.1 reasoning
)
```

**Response Parsing**:
```python
# Multiple parsing strategies for robustness
patterns = [
    r'(\d+)\.\s*(yes|no)',           # "1. yes"
    r'(\d+)\s*[-:]\s*(yes|no)',      # "1: yes" or "1 - yes"
    r'Question\s*(\d+)\s*[-:]\s*(yes|no)',
    r'Q(\d+)\s*[-:]\s*(yes|no)',
]
```

##### `compute_text3d_for_asset(...)`
Main entry point for Text-3D alignment metric.

**Flow**:
1. Load questions from `questions/questions.json`
2. Extract 12 frames from video → save to `vqa_frames/`
3. For each frame: call GPT-5.1 VQA
4. Aggregate answers via majority voting
5. Save results with vote breakdown

**Output Structure** (`text3d_results.json`):
```json
{
  "questions": { "1": "...", "2": "...", ... },
  "final_answers": { "1": "yes", "2": "no", ... },
  "vote_breakdown": {
    "1": { "yes": 8, "no": 4, "final": "yes" }
  },
  "per_frame_answers": [
    { "1": "yes", "2": "yes", ... },
    ...
  ],
  "score": 60.0,
  "frames_dir": "...",
  "frame_paths": [...]
}
```

---

#### Suggestions Pipeline

##### `_generate_prompt_based_questions(...)`
```python
def _generate_prompt_based_questions(
    prompt: str, 
    api_key: str, 
    n_questions: int = 3
) -> List[str]:
```
- Uses GPT-5.1 to generate prompt-specific questions
- Returns list of open-ended evaluation questions

##### `_call_gpt4o_detailed_analysis(...)`
```python
def _call_gpt4o_detailed_analysis(
    frame_paths: List[Path],
    prompt: str,
    custom_questions: List[str],
    api_key: str,
    analysis_type: str = "comprehensive"
) -> Dict[str, Any]:
```

**Analysis Prompt Structure**:
```
You are analyzing a 3D model from multiple viewpoints...

## 1. GEOMETRY FLAWS
**Score:** X/10
**Key Issues:**
- ...
**Suggestions:**
- ...

## 2. TEXTURE & MATERIAL FLAWS
...

## 3. MULTI-VIEW CONSISTENCY
...

## 4. SEMANTIC REASONABLENESS
...

## 5. PROMPT-SPECIFIC EVALUATION
### Q1
**Score:** X/10
**Findings:**
- ...
**Suggestion:** ...
```

##### `_parse_analysis_sections(...)`
Parses GPT-5.1 response into structured dictionary.

**Section Markers** (regex):
```python
section_markers = [
    (r"##?\s*1\.?\s*GEOMETRY", "geometry"),
    (r"##?\s*2\.?\s*TEXTURE", "texture"),
    (r"##?\s*3\.?\s*MULTI[\-\u2011]?VIEW|##?\s*3\.?\s*CONSISTENCY", "consistency"),
    (r"##?\s*4\.?\s*SEMANTIC|##?\s*4\.?\s*REASONABLENESS", "reasonableness"),
    (r"##?\s*5\.?\s*PROMPT", "prompt_specific"),
    (r"##?\s*OVERALL", "overall"),
]
```

**Score Extraction Patterns**:
```python
score_patterns = [
    r"\*\*(?:Overall\s+Quality\s+)?Score[:\s]*(\d+(?:\.\d+)?)\s*/\s*10\*\*",
    r"\*\*Score:\*\*\s*(\d+(?:\.\d+)?)\s*/\s*10",
    r"Score[:\s]*(\d+(?:\.\d+)?)\s*/\s*10",
    r"(\d+(?:\.\d+)?)\s*/\s*10",
]
```

**Issues/Suggestions Markers**:
```python
issues_marker = r"\*\*Key\s+Issues:\*\*"
suggestions_marker = r"\*\*Suggestions:\*\*"
```

##### `compute_text3d_suggestions(...)`
Main entry point for suggestions pipeline.

**Flow**:
1. Check preconditions (video exists, API key set)
2. Extract frames → save to `suggestion_frames/`
3. Generate prompt-specific questions
4. Call detailed analysis with all frames
5. Parse response into structured sections
6. Save to `text3d_suggestions.json`

**Output Structure** (`text3d_suggestions.json`):
```json
{
  "prompt": "...",
  "n_frames_analyzed": 12,
  "custom_questions": ["Q1...", "Q2...", "Q3..."],
  "sections": {
    "geometry": {
      "title": "Geometry Flaws",
      "score": 7.5,
      "key_issues": ["issue1", "issue2"],
      "suggestions": ["suggestion1", "suggestion2"]
    },
    "texture": { ... },
    "consistency": { ... },
    "reasonableness": { ... },
    "prompt_specific": {
      "title": "Prompt-Specific Evaluation",
      "score": 6.3,  // Average of Q scores
      "key_issues": [...],
      "suggestions": [...],
      "questions": [
        {
          "number": 1,
          "title": "Question text...",
          "score": 7,
          "findings": ["finding1", "finding2"],
          "suggestion": "..."
        },
        ...
      ]
    },
    "overall": { ... }
  },
  "raw_analysis": "Full GPT response..."
}
```

---

### 5. `metrics/geometric.py` - Geometric Consistency

**Dependencies**: torch, DepthAnything model

**Key Functions**:

- `_load_depth_anything_model()` - Loads ViT-L depth estimation model
- `_run_depth_anything_on_images()` - Estimates depth maps
- `_compute_geometric_metric()` - Compares rendered normals vs depth-estimated normals

**Metric Calculation**:
1. Load rendered RGB images and world-space normals
2. Estimate depth using DepthAnything
3. Compute normals from depth gradient
4. Compare rendered vs estimated normals
5. Return consistency score

---

### 6. `metrics/semantic.py` - Semantic Consistency

**Dependencies**: torch, DINO model

Uses DINO features projected onto mesh vertices across views to measure semantic consistency.

---

### 7. `metrics/aesthetics.py` - Aesthetics Score

**Dependencies**: ImageReward model

```python
def compute_aesthetics_for_video(video_path: Path, prompt: str = "a 3D render") -> Optional[float]:
```

- Extracts frames from video
- Scores each frame with ImageReward
- Returns average score

---

## OpenAI API Usage Notes

### GPT-5.1 Specific Requirements

```python
# CORRECT for GPT-5.1
response = client.chat.completions.create(
    model="gpt-5.1",
    messages=[...],
    max_completion_tokens=4000,  # NOT max_tokens
    reasoning_effort="medium",    # Optional: low/medium/high
    # NOTE: temperature NOT supported with reasoning
)

# WRONG - will cause 400 error
response = client.chat.completions.create(
    model="gpt-5.1",
    messages=[...],
    max_tokens=4000,        # ❌ Not supported
    temperature=0.7,        # ❌ Not supported with reasoning
)
```

### Vision API Format

```python
{
    "role": "user",
    "content": [
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_image}",
                "detail": "low"  # or "high" for suggestions
            }
        },
        {"type": "text", "text": prompt}
    ]
}
```

---

## Data Flow Diagram

```
mesh.obj + prompt
     │
     ▼
┌─────────────────────┐
│  render_asset.py    │
│  - render video     │
│  - render 120 views │
│  - generate batch   │
└──────────┬──────────┘
           │
           ▼
    asset_folder/
    ├── video/turntable.mp4
    ├── save/it0-test/...
    └── questions/questions.json
           │
     ┌─────┴─────┐
     ▼           ▼
┌─────────┐  ┌──────────────┐
│ text3d  │  │  suggestions │
│  VQA    │  │   analysis   │
└────┬────┘  └──────┬───────┘
     │              │
     ▼              ▼
text3d_results.json  text3d_suggestions.json
```

---

## Testing Commands

```bash
# Run text3d metric on existing asset
uv run python -c "
from pathlib import Path
from eval3d_pipeline.metrics.text3d import compute_text3d_for_asset
from eval3d_pipeline.config import Settings

asset_folder = Path('data/my_algo/_1209214233_texture')
settings = Settings()
score = compute_text3d_for_asset(asset_folder, settings=settings)
"

# Run suggestions on existing asset
uv run eval3d-pipeline suggest data/my_algo/_1209214233_texture \
    --prompt "Full-body male sci-fi character" \
    --frames 12

# Re-render video with new settings
uv run eval3d-pipeline render-video data/my_algo/asset/model.glb \
    -o data/my_algo/asset/video/turntable.mp4 \
    --size 1024 --distance 1.8 --elevation 15
```

---

## Common Issues & Solutions

### 1. Missing Textures (Clay Render)
**Cause**: OBJ file references `mtllib file.mtl` but MTL file is missing
**Solution**: Ensure MTL file and texture images are in the same directory as OBJ

### 2. API Error: max_tokens not supported
**Cause**: GPT-5.1 requires `max_completion_tokens`
**Solution**: Replace `max_tokens` with `max_completion_tokens`

### 3. API Error: temperature not supported
**Cause**: GPT-5.1 with reasoning doesn't support temperature
**Solution**: Remove `temperature` parameter from API call

### 4. Empty sections in suggestions JSON
**Cause**: Parsing regex doesn't match model output format
**Solution**: Update regex patterns in `_parse_analysis_sections()`

### 5. Low VQA score despite good render
**Cause**: Majority voting across frames - some angles may fail
**Solution**: Check `per_frame_answers` in results to see which frames/questions failed

---

## Version History

### Dec 2024
- Upgraded VQA from GPT-4o to GPT-5.1
- Added comprehensive suggestions pipeline
- Increased video resolution: 512 → 1024
- Improved texture detection for rendering
- Added per-frame answer logging
- Added vote breakdown in text3d results
- Fixed API compatibility for GPT-5.1 (max_completion_tokens, no temperature)


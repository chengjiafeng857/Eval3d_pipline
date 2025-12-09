from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
from rich.console import Console

from ..config import Settings, get_settings

console = Console()

# Lazy imports
_CV2_AVAILABLE = False
_PIL_AVAILABLE = False

try:
    import cv2
    _CV2_AVAILABLE = True
except ImportError:
    pass

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    pass


def _extract_key_frames(video_path: Path, n_frames: int = 12) -> List[Path]:
    """Extract key frames from video for VQA."""
    if not _CV2_AVAILABLE:
        raise ImportError("opencv-python required")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    
    temp_dir = Path(tempfile.mkdtemp())
    frame_paths = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = temp_dir / f"{i}.png"
            cv2.imwrite(str(frame_path), frame)
            frame_paths.append(frame_path)
    
    cap.release()
    return frame_paths


def _encode_image_to_base64(image_path: Path) -> str:
    """Encode an image to base64 for OpenAI API."""
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow required")
    
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    # Resize for efficiency
    img.thumbnail((512, 512))
    
    temp_path = Path(tempfile.mktemp(suffix=".jpg"))
    img.save(temp_path)
    
    with open(temp_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    
    temp_path.unlink()
    return data


def _call_gpt4o_vqa(image_path: Path, questions: Dict[str, str], api_key: str) -> Dict[int, str]:
    """Call GPT-4o for VQA on a single image."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    client = OpenAI(api_key=api_key)
    
    # Encode image
    b64_image = _encode_image_to_base64(image_path)
    
    # Build prompt
    questions_text = "\n".join([f"Q[{k}]: {v}" for k, v in questions.items()])
    
    prompt = f"""Given an image and multiple questions, answer the questions with yes or no based on the image. Follow the formatting examples below.

Formatting example:
Questions:
Q[1]: Is there a cat?
Q[2]: Is the cat black?

Answers:
Q[1]: Is there a cat?
A[1]: Yes
Q[2]: Is the cat black?
A[2]: No

Now answer the following questions based on the image:
Questions:
{questions_text}

Answers:
"""
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "low"}
                    },
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        max_tokens=500,
    )
    
    # Parse response
    answer_text = response.choices[0].message.content
    answers = {}
    
    # Try multiple parsing strategies
    import re
    
    # Strategy 1: Look for A[n]: pattern
    for line in answer_text.split("\n"):
        line = line.strip()
        if line.startswith("A["):
            try:
                match = re.match(r"A\[(\d+)\]:\s*(.*)", line)
                if match:
                    q_num = int(match.group(1))
                    answer = match.group(2).strip().lower()
                    answers[q_num] = "yes" if "yes" in answer else "no"
            except (ValueError, IndexError):
                continue
    
    # Strategy 2: Look for "1." or "Q1:" patterns if strategy 1 failed
    if not answers:
        for line in answer_text.split("\n"):
            line = line.strip()
            # Match patterns like "1. Yes" or "Q1: Yes" or "1: Yes"
            match = re.match(r"(?:Q)?(\d+)[.:\)]\s*(yes|no)", line, re.IGNORECASE)
            if match:
                q_num = int(match.group(1))
                answer = match.group(2).lower()
                answers[q_num] = answer
    
    # Strategy 3: Just look for yes/no in each line with a number
    if not answers:
        lines = answer_text.split("\n")
        for line in lines:
            line = line.strip().lower()
            for q_num in range(1, 6):
                if str(q_num) in line and ("yes" in line or "no" in line):
                    answers[q_num] = "yes" if "yes" in line else "no"
                    break
    
    return answers


def compute_text3d_for_asset(
    asset_folder: Path,
    questions_json: Optional[Path] = None,
    settings: Optional[Settings] = None,
) -> Optional[float]:
    """
    Run text-3D alignment metric using GPT-4o VQA.
    
    This implements the Eval3D text-3D alignment directly without
    requiring the notebook.
    """
    settings = settings or get_settings()
    
    video_path = asset_folder / "video" / "turntable.mp4"
    if not video_path.exists():
        console.print(f"[yellow]Skipping text3d (video missing)[/yellow] in {asset_folder}")
        return None

    q_path = questions_json or (asset_folder / "questions" / "questions.json")
    if not q_path.exists():
        console.print(f"[yellow]Skipping text3d (questions missing)[/yellow] in {asset_folder}")
        return None
    
    if not settings.openai_api_key:
        console.print("[yellow]Skipping text3d (OPENAI_API_KEY not set)[/yellow]")
        return None
    
    try:
        # Load questions
        with open(q_path) as f:
            questions = json.load(f)
        
        console.print(f"[blue]Extracting frames for VQA...[/blue]")
        frame_paths = _extract_key_frames(video_path, n_frames=12)
        
        if not frame_paths:
            console.print("[red]No frames extracted[/red]")
            return None
        
        # Run VQA on multiple frames and aggregate
        console.print(f"[blue]Running GPT-4o VQA on {len(frame_paths)} frames...[/blue]")
        
        all_answers: Dict[int, List[str]] = {int(k): [] for k in questions.keys()}
        
        for i, frame_path in enumerate(frame_paths):
            console.print(f"  Processing frame {i+1}/{len(frame_paths)}...")
            try:
                answers = _call_gpt4o_vqa(frame_path, questions, settings.openai_api_key)
                if i == 0:  # Debug: show first frame's response
                    console.print(f"  [dim]Frame 0 answers: {answers}[/dim]")
                for q_num, answer in answers.items():
                    if q_num in all_answers:
                        all_answers[q_num].append(answer)
            except Exception as e:
                console.print(f"[yellow]VQA failed for frame {i}: {e}[/yellow]")
                import traceback
                traceback.print_exc()
        
        # Aggregate answers (majority vote)
        final_answers = {}
        for q_num, answers_list in all_answers.items():
            if answers_list:
                yes_count = sum(1 for a in answers_list if a == "yes")
                final_answers[q_num] = "yes" if yes_count > len(answers_list) / 2 else "no"
        
        # Compute score (% of "yes" answers, assuming questions are positive)
        if final_answers:
            yes_count = sum(1 for a in final_answers.values() if a == "yes")
            score = yes_count / len(final_answers) * 100
        else:
            score = 0.0
        
        # Save results
        results_path = asset_folder / "text3d_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "questions": questions,
                "answers": final_answers,
                "score": score
            }, f, indent=2)
        
        # Cleanup
        for p in frame_paths:
            p.unlink(missing_ok=True)
        if frame_paths:
            frame_paths[0].parent.rmdir()
        
        console.print(f"[green]Text-3D alignment score: {score:.1f}%[/green]")
        console.print(f"[dim]Results saved to {results_path}[/dim]")
        
        return score
        
    except ImportError as e:
        console.print(f"[red]Missing dependency[/red]: {e}")
        return None
    except Exception as e:
        console.print(f"[red]Text-3D alignment failed[/red]: {e}")
        import traceback
        traceback.print_exc()
        return None


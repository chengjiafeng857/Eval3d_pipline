from __future__ import annotations

import base64
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
from rich.console import Console

from ..config import Settings, get_settings

console = Console()

# Model selection - use latest GPT-5.1 for best vision + reasoning capabilities
# GPT-5.1 is OpenAI's newest flagship with 400K context, configurable reasoning
GPT_MODEL = "gpt-5.1"
GPT_MODEL_MINI = "gpt-5-mini"
GPT_MODEL_FAST = "gpt-5-nano"  # For high-throughput simple tasks

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


def _extract_key_frames(
    video_path: Path, 
    n_frames: int = 12,
    output_dir: Optional[Path] = None,
    prefix: str = "frame"
) -> List[Path]:
    """Extract key frames from video for VQA.
    
    Args:
        video_path: Path to video file
        n_frames: Number of frames to extract
        output_dir: Directory to save frames (uses temp dir if None)
        prefix: Prefix for frame filenames
    
    Returns:
        List of paths to extracted frame images
    """
    if not _CV2_AVAILABLE:
        raise ImportError("opencv-python required")
    
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
    
    # Use provided output_dir or create temp directory
    if output_dir:
        save_dir = output_dir
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(tempfile.mkdtemp())
    
    frame_paths = []
    
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_path = save_dir / f"{prefix}_{i:04d}.png"
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
        model=GPT_MODEL,  # Using GPT-5.1 for better VQA accuracy
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
        max_completion_tokens=1500,  # Increased to handle longer questions with detailed prompts
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
        
        # Extract frames - save to asset folder for inspection
        vqa_frames_dir = asset_folder / "vqa_frames"
        console.print(f"[blue]Extracting frames for VQA...[/blue]")
        console.print(f"[dim]   Saving to: {vqa_frames_dir}[/dim]")
        
        frame_paths = _extract_key_frames(
            video_path, 
            n_frames=12,
            output_dir=vqa_frames_dir,
            prefix="vqa"
        )
        
        if not frame_paths:
            console.print("[red]No frames extracted[/red]")
            return None
        
        # Run VQA on multiple frames and aggregate
        console.print(f"[blue]Running VQA ({GPT_MODEL}) on {len(frame_paths)} frames...[/blue]")
        
        all_answers: Dict[int, List[str]] = {int(k): [] for k in questions.keys()}
        frame_answers_log: List[Dict[int, str]] = []  # Store all frames' answers for debugging
        
        for i, frame_path in enumerate(frame_paths):
            console.print(f"  Processing frame {i+1}/{len(frame_paths)}...")
            try:
                answers = _call_gpt4o_vqa(frame_path, questions, settings.openai_api_key)
                frame_answers_log.append(answers)
                # Show answers for each frame
                ans_str = ", ".join([f"Q{k}:{v}" for k, v in sorted(answers.items())])
                console.print(f"    [dim]→ {ans_str}[/dim]")
                for q_num, answer in answers.items():
                    if q_num in all_answers:
                        all_answers[q_num].append(answer)
            except Exception as e:
                console.print(f"[yellow]VQA failed for frame {i}: {e}[/yellow]")
                frame_answers_log.append({})
                import traceback
                traceback.print_exc()
        
        # Aggregate answers (majority vote) and show vote breakdown
        console.print("\n[blue]Vote breakdown by question:[/blue]")
        final_answers = {}
        for q_num, answers_list in all_answers.items():
            if answers_list:
                yes_count = sum(1 for a in answers_list if a == "yes")
                no_count = len(answers_list) - yes_count
                final = "yes" if yes_count > len(answers_list) / 2 else "no"
                final_answers[q_num] = final
                # Show vote breakdown
                q_text = str(questions.get(str(q_num), questions.get(q_num, "")))[:50]
                console.print(f"  Q{q_num}: [green]{yes_count} yes[/green] / [red]{no_count} no[/red] → [bold]{final}[/bold]")
        
        # Compute score (% of "yes" answers, assuming questions are positive)
        if final_answers:
            yes_count = sum(1 for a in final_answers.values() if a == "yes")
            score = yes_count / len(final_answers) * 100
        else:
            score = 0.0
        
        # Build vote breakdown for each question
        vote_breakdown = {}
        for q_num, answers_list in all_answers.items():
            if answers_list:
                yes_count = sum(1 for a in answers_list if a == "yes")
                vote_breakdown[q_num] = {
                    "yes": yes_count,
                    "no": len(answers_list) - yes_count,
                    "final": final_answers.get(q_num, "unknown")
                }
        
        # Save results
        results_path = asset_folder / "text3d_results.json"
        with open(results_path, "w") as f:
            json.dump({
                "questions": questions,
                "final_answers": final_answers,
                "vote_breakdown": vote_breakdown,
                "per_frame_answers": frame_answers_log,
                "score": score,
                "frames_dir": str(vqa_frames_dir),
                "frame_paths": [str(p) for p in frame_paths],
            }, f, indent=2)
        
        # Note: frames are kept in the asset folder for inspection
        # They are saved to: {asset_folder}/vqa_frames/
        
        console.print(f"[green]Text-3D alignment score: {score:.1f}%[/green]")
        console.print(f"[dim]Results saved to {results_path}[/dim]")
        console.print(f"[dim]Frames saved to {vqa_frames_dir}/ (kept for inspection)[/dim]")
        
        return score
        
    except ImportError as e:
        console.print(f"[red]Missing dependency[/red]: {e}")
        return None
    except Exception as e:
        console.print(f"[red]Text-3D alignment failed[/red]: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# NEW: GPT-4o Suggestion Pipeline for 3D Model Quality Analysis
# =============================================================================

def _generate_prompt_based_questions(prompt: str, api_key: str, n_questions: int = 3) -> List[str]:
    """
    Generate open-ended questions specific to the prompt using GPT-4o.
    
    These questions are tailored to the specific 3D asset being evaluated.
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    client = OpenAI(api_key=api_key)
    
    system_prompt = """You are an expert 3D artist and quality assurance specialist. 
Given a text prompt describing a 3D model, generate specific questions that would help 
evaluate if the generated 3D model accurately represents the prompt.

Focus on:
- Specific features mentioned in the prompt
- Expected proportions and spatial relationships
- Material/texture expectations based on the description
- Pose, action, or state if mentioned
- Context-specific details that should be visible

Return ONLY the questions, one per line, numbered 1-N. No explanations."""

    user_prompt = f"""Generate {n_questions} specific evaluation questions for a 3D model created from this prompt:

"{prompt}"

These questions should help identify if the 3D model correctly captures the essence of the prompt."""

    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=500,  # GPT-5 uses max_completion_tokens
        # Note: temperature not supported with reasoning_effort in GPT-5
        reasoning_effort="low",  # GPT-5.1: use light reasoning for question generation
    )
    
    # Parse questions from response
    raw_text = response.choices[0].message.content
    questions = []
    for line in raw_text.strip().split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove numbering like "1.", "1)", "1:"
            import re
            cleaned = re.sub(r"^\d+[\.\)\:\-]\s*", "", line)
            if cleaned:
                questions.append(cleaned)
        elif line and not line[0].isdigit():
            questions.append(line)
    
    return questions[:n_questions]


def _encode_image_high_quality(image_path: Path, max_size: int = 1024) -> str:
    """Encode image at higher quality for detailed analysis."""
    if not _PIL_AVAILABLE:
        raise ImportError("Pillow required")
    
    img = Image.open(image_path)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    
    # Use higher resolution for better analysis
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    
    temp_path = Path(tempfile.mktemp(suffix=".jpg"))
    img.save(temp_path, quality=90)
    
    with open(temp_path, "rb") as f:
        data = base64.b64encode(f.read()).decode("utf-8")
    
    temp_path.unlink()
    return data


def _call_gpt4o_detailed_analysis(
    frame_paths: List[Path],
    prompt: str,
    custom_questions: List[str],
    api_key: str,
    analysis_type: str = "comprehensive"
) -> Dict[str, Any]:
    """
    Call GPT-4o for detailed 3D model analysis with multiple frames.
    
    Args:
        frame_paths: List of frame image paths from different angles
        prompt: Original text prompt used to generate the 3D model
        custom_questions: Additional prompt-specific questions
        api_key: OpenAI API key
        analysis_type: Type of analysis to perform
    
    Returns:
        Dictionary containing detailed analysis results
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")
    
    client = OpenAI(api_key=api_key)
    
    # Encode multiple frames (select key angles)
    n_frames_to_use = min(6, len(frame_paths))
    selected_indices = np.linspace(0, len(frame_paths) - 1, n_frames_to_use, dtype=int)
    
    image_contents = []
    for idx in selected_indices:
        b64_image = _encode_image_high_quality(frame_paths[idx])
        image_contents.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_image}", "detail": "high"}
        })
    
    # Build comprehensive analysis prompt with strict format for prompt-specific section
    # Format custom questions with Q1, Q2, Q3 labels for consistent parsing
    custom_q_formatted = []
    for i, q in enumerate(custom_questions, 1):
        custom_q_formatted.append(f"Q{i}: {q}")
    custom_questions_text = "\n".join(custom_q_formatted) if custom_q_formatted else "Q1: Does the model match the prompt?"
    
    analysis_prompt = f"""You are an expert 3D artist, technical director, and quality assurance specialist reviewing a 3D model.
The model was generated from this text prompt: "{prompt}"

I'm showing you {n_frames_to_use} rendered views of the 3D model from different angles (turntable rotation).

Please provide a DETAILED and CONSTRUCTIVE analysis covering these 5 areas:

## 1. GEOMETRY FLAWS
Analyze the model's geometric quality:
- Overall mesh quality (smoothness, topology issues, artifacts)
- Specific problematic areas (floating geometry, holes, self-intersections)
- Proportions and anatomical/structural accuracy
- Level of detail consistency across the model
- Any missing or malformed parts

## 2. TEXTURE & MATERIAL FLAWS  
Analyze the texturing and materials:
- Overall texture quality (resolution, sharpness, coherence)
- UV mapping issues (stretching, seams, distortion)
- Material consistency (does the surface look appropriate?)
- Color bleeding or incorrect color regions
- Missing or incorrect textures on specific parts

## 3. MULTI-VIEW CONSISTENCY
Analyze consistency across different viewing angles:
- Does the model look coherent from all angles?
- Are there view-dependent artifacts or "Janus face" issues?
- Backside quality compared to front
- Any parts that only look correct from certain angles?
- Lighting/shading consistency

## 4. SEMANTIC REASONABLENESS
Analyze if the model makes sense to a human viewer:
- Does it accurately represent the prompt?
- Are there any physically impossible or nonsensical elements?
- Do proportions match real-world expectations?
- Is the pose/configuration natural and believable?
- Any uncanny valley effects or disturbing artifacts?

## 5. PROMPT-SPECIFIC EVALUATION
Answer EACH of these questions using EXACTLY this format:

{custom_questions_text}

For EACH question above, you MUST use this EXACT format:
### Q1
**Score:** X/10
**Findings:**
- [finding 1]
- [finding 2]
**Suggestion:** [your suggestion]

### Q2
**Score:** X/10
**Findings:**
- [finding 1]
- [finding 2]
**Suggestion:** [your suggestion]

### Q3
**Score:** X/10
**Findings:**
- [finding 1]
- [finding 2]
**Suggestion:** [your suggestion]

---

For sections 1-4, provide:
1. **Score:** X/10
2. **Key Issues** (bullet list of specific problems found)
3. **Suggestions** (how to fix or improve)

Be specific and reference particular parts of the model when possible (e.g., "the left arm", "the back of the head", "the base/pedestal").

End with an **OVERALL ASSESSMENT** including:
- Overall quality score (1-10)
- Top 3 most critical issues to fix
- General recommendations for improvement
"""

    # Build message with images
    content = image_contents + [{"type": "text", "text": analysis_prompt}]
    
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {
                "role": "user",
                "content": content
            }
        ],
        max_completion_tokens=4000,  # GPT-5 uses max_completion_tokens
        # Note: temperature not supported with reasoning_effort in GPT-5
        reasoning_effort="medium",  # GPT-5.1: use medium reasoning for thorough analysis
    )
    
    analysis_text = response.choices[0].message.content
    
    # Parse the response into structured sections
    result = {
        "raw_analysis": analysis_text,
        "prompt": prompt,
        "n_frames_analyzed": n_frames_to_use,
        "custom_questions": custom_questions,
        "sections": _parse_analysis_sections(analysis_text, custom_questions),
    }
    
    return result


def _parse_analysis_sections(analysis_text: str, custom_questions: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """Parse the GPT response into structured sections with score, key_issues, and suggestions."""
    import re
    custom_questions = custom_questions or []
    
    sections = {
        "geometry": {"title": "Geometry Flaws", "score": None, "key_issues": [], "suggestions": []},
        "texture": {"title": "Texture & Material Flaws", "score": None, "key_issues": [], "suggestions": []},
        "consistency": {"title": "Multi-View Consistency", "score": None, "key_issues": [], "suggestions": []},
        "reasonableness": {"title": "Semantic Reasonableness", "score": None, "key_issues": [], "suggestions": []},
        "prompt_specific": {"title": "Prompt-Specific Evaluation", "score": None, "key_issues": [], "suggestions": [], "questions": []},
        "overall": {"title": "Overall Assessment", "score": None, "key_issues": [], "suggestions": []},
    }
    
    # Score patterns
    score_patterns = [
        r"\*\*(?:Overall\s+Quality\s+)?Score[:\s]*(\d+(?:\.\d+)?)\s*/\s*10\*\*",
        r"\*\*Score\*\*[:\s]*(\d+(?:\.\d+)?)",
        r"Score[:\s]*(\d+(?:\.\d+)?)\s*/\s*10",
        r"Score[:\s]*(\d+(?:\.\d+)?)\s*out\s*of\s*10",
        r"(\d+(?:\.\d+)?)\s*/\s*10",
    ]
    
    # Section markers
    section_markers = [
        (r"##?\s*1\.?\s*GEOMETRY", "geometry"),
        (r"##?\s*2\.?\s*TEXTURE", "texture"),
        (r"##?\s*3\.?\s*MULTI[\-\u2011]?VIEW|##?\s*3\.?\s*CONSISTENCY", "consistency"),
        (r"##?\s*4\.?\s*SEMANTIC|##?\s*4\.?\s*REASONABLENESS", "reasonableness"),
        (r"##?\s*5\.?\s*PROMPT", "prompt_specific"),
        (r"##?\s*OVERALL", "overall"),
    ]
    
    # Subsection markers - match both markdown headers and bold text formats (with optional colon)
    issues_marker = r"###?\s*(?:Key\s+)?Issues|\*\*(?:Key\s+)?Issues:?\*\*|Top\s+\d+\s+Critical|\*\*Top\s+\d+\s+(?:Critical|most\s+critical)"
    suggestions_marker = r"###?\s*Suggestions|\*\*Suggestions:?\*\*|###?\s*General\s+[Rr]ecommendations|\*\*General\s+[Rr]ecommendations:?\*\*"
    findings_marker = r"###?\s*\d+\)|Findings|Assessment"
    
    current_section = None
    current_subsection = None  # "issues", "suggestions", "findings", or None
    current_prompt_question = None  # For tracking prompt-specific questions
    prompt_question_scores = []  # Collect scores for prompt-specific questions
    lines = analysis_text.split("\n")
    
    for line in lines:
        # Check for main section headers
        for pattern, section_key in section_markers:
            if re.search(pattern, line, re.IGNORECASE):
                # Finalize any current prompt question before switching sections
                if current_prompt_question:
                    sections["prompt_specific"]["questions"].append(current_prompt_question)
                    current_prompt_question = None
                current_section = section_key
                current_subsection = None
                break
        
        if not current_section:
            continue
        
        # Special handling for prompt_specific section - strict format: ### Q1, ### Q2, ### Q3
        if current_section == "prompt_specific":
            line_stripped = line.strip()
            
            # Skip main section header (## 5. PROMPT)
            if re.match(r"^##\s*\d+\.\s*\w", line_stripped):
                continue
            
            # Check for question headers: ### Q1, ### Q2, ### Q3 (strict format)
            q_match = re.match(r"^###\s*Q(\d+)", line_stripped)
            if q_match:
                # Finalize previous question
                if current_prompt_question:
                    sections["prompt_specific"]["questions"].append(current_prompt_question)
                
                q_num = int(q_match.group(1))
                # Get question title from custom_questions
                q_title = custom_questions[q_num - 1] if q_num <= len(custom_questions) else f"Question {q_num}"
                current_prompt_question = {
                    "number": q_num, 
                    "title": q_title, 
                    "score": None, 
                    "findings": [], 
                    "suggestion": ""
                }
                current_subsection = None
                continue
            
            # Check for **Score:** line
            score_match = re.match(r"^\*\*Score:\*\*\s*(\d+(?:\.\d+)?)\s*/\s*10", line_stripped, re.IGNORECASE)
            if score_match and current_prompt_question:
                try:
                    score = float(score_match.group(1))
                    if 0 <= score <= 10:
                        current_prompt_question["score"] = score
                        prompt_question_scores.append(score)
                except ValueError:
                    pass
                continue
            
            # Check for **Findings:** marker
            if re.match(r"^\*\*Findings:\*\*", line_stripped, re.IGNORECASE):
                current_subsection = "q_findings"
                continue
            
            # Check for **Suggestion:** marker (with inline text or on next line)
            sugg_match = re.match(r"^\*\*Suggestion:\*\*\s*(.*)", line_stripped, re.IGNORECASE)
            if sugg_match and current_prompt_question:
                current_subsection = "q_suggestion"
                sugg_text = sugg_match.group(1).strip()
                if sugg_text:
                    current_prompt_question["suggestion"] = sugg_text
                continue
            
            # Extract bullet points for findings
            if current_prompt_question and current_subsection == "q_findings":
                if line_stripped.startswith("- ") or line_stripped.startswith("* "):
                    bullet_text = line_stripped[2:].strip()
                    bullet_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", bullet_text)
                    if bullet_text:
                        current_prompt_question["findings"].append(bullet_text)
                    continue
            
            # Extract suggestion (continuation or bullet)
            if current_prompt_question and current_subsection == "q_suggestion":
                if line_stripped.startswith("- "):
                    bullet_text = line_stripped[2:].strip()
                    if current_prompt_question["suggestion"]:
                        current_prompt_question["suggestion"] += " " + bullet_text
                    else:
                        current_prompt_question["suggestion"] = bullet_text
                elif line_stripped and not line_stripped.startswith("#") and not line_stripped.startswith("**") and not line_stripped.startswith("---"):
                    if current_prompt_question["suggestion"]:
                        current_prompt_question["suggestion"] += " " + line_stripped
                    else:
                        current_prompt_question["suggestion"] = line_stripped
                continue
            
            # Check if we hit section end
            if line_stripped.startswith("---") or (line_stripped.startswith("##") and not line_stripped.startswith("###")):
                if current_prompt_question:
                    sections["prompt_specific"]["questions"].append(current_prompt_question)
                    current_prompt_question = None
            
            continue
            
        # Check for subsection headers (for non-prompt_specific sections)
        if re.search(issues_marker, line, re.IGNORECASE):
            current_subsection = "issues"
            continue
        elif re.search(suggestions_marker, line, re.IGNORECASE):
            current_subsection = "suggestions"
            continue
        
        # Extract score from this line (for main sections)
        if sections[current_section]["score"] is None:
            for score_pattern in score_patterns:
                match = re.search(score_pattern, line, re.IGNORECASE)
                if match:
                    try:
                        score = float(match.group(1))
                        if 0 <= score <= 10:
                            sections[current_section]["score"] = score
                            break
                    except ValueError:
                        pass
        
        # Extract bullet points for issues/suggestions
        line_stripped = line.strip()
        if line_stripped.startswith("- ") or line_stripped.startswith("* "):
            # Clean up the bullet point
            bullet_text = line_stripped[2:].strip()
            # Remove markdown bold markers
            bullet_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", bullet_text)
            
            if bullet_text:
                if current_subsection == "issues":
                    sections[current_section]["key_issues"].append(bullet_text)
                elif current_subsection == "suggestions":
                    sections[current_section]["suggestions"].append(bullet_text)
        
        # Also capture numbered items (1., 2., etc.) for overall section
        elif re.match(r"^\d+\.\s+", line_stripped):
            bullet_text = re.sub(r"^\d+\.\s+", "", line_stripped)
            bullet_text = re.sub(r"\*\*([^*]+)\*\*", r"\1", bullet_text)
            if bullet_text and current_section == "overall":
                if current_subsection == "issues":
                    sections[current_section]["key_issues"].append(bullet_text)
                elif current_subsection == "suggestions":
                    sections[current_section]["suggestions"].append(bullet_text)
    
    # Finalize any remaining prompt question
    if current_prompt_question:
        sections["prompt_specific"]["questions"].append(current_prompt_question)
    
    # Build key_issues and suggestions from parsed prompt_specific questions
    for q in sections["prompt_specific"]["questions"]:
        q_num = q.get("number", "?")
        q_title = q.get("title", "")
        q_score = q.get("score")
        
        # Add to key_issues
        summary = f"Q{q_num}: {q_title}"
        if q_score is not None:
            summary += f" (Score: {q_score}/10)"
        sections["prompt_specific"]["key_issues"].append(summary)
        
        # Add findings to suggestions
        for finding in q.get("findings", []):
            sections["prompt_specific"]["suggestions"].append(f"Q{q_num} Finding: {finding}")
        
        # Add question suggestion
        if q.get("suggestion"):
            sections["prompt_specific"]["suggestions"].append(f"Q{q_num} Suggestion: {q['suggestion']}")
    
    # Calculate average score for prompt_specific if we found individual question scores
    if prompt_question_scores and sections["prompt_specific"]["score"] is None:
        sections["prompt_specific"]["score"] = sum(prompt_question_scores) / len(prompt_question_scores)
    
    return sections


def compute_text3d_suggestions(
    asset_folder: Path,
    prompt: Optional[str] = None,
    settings: Optional[Settings] = None,
    n_frames: int = 12,
) -> Optional[Dict[str, Any]]:
    """
    Run comprehensive 3D model quality analysis using GPT-4o.
    
    This provides detailed suggestions and feedback on:
    1. Geometry flaws (general and specific parts)
    2. Texture flaws (general and specific parts)
    3. Multi-view consistency issues
    4. Semantic reasonableness
    5. Prompt-specific evaluation questions (auto-generated)
    
    Args:
        asset_folder: Path to the asset folder containing video/turntable.mp4
        prompt: Text prompt used to generate the 3D model (optional but recommended)
        settings: Pipeline settings (will use defaults if not provided)
        n_frames: Number of frames to extract from video
    
    Returns:
        Dictionary containing detailed analysis results, or None on failure
    """
    settings = settings or get_settings()
    
    video_path = asset_folder / "video" / "turntable.mp4"
    if not video_path.exists():
        console.print(f"[yellow]Skipping suggestions (video missing)[/yellow] in {asset_folder}")
        return None
    
    if not settings.openai_api_key:
        console.print("[yellow]Skipping suggestions (OPENAI_API_KEY not set)[/yellow]")
        return None
    
    # Try to load prompt from questions.json if not provided
    if not prompt:
        q_path = asset_folder / "questions" / "questions.json"
        if q_path.exists():
            with open(q_path) as f:
                data = json.load(f)
                if isinstance(data, dict) and "prompt" in data:
                    prompt = data["prompt"]
        
        # Also try prompt.txt
        prompt_path = asset_folder / "prompt.txt"
        if not prompt and prompt_path.exists():
            prompt = prompt_path.read_text().strip()
        
        if not prompt:
            prompt = asset_folder.name.replace("_", " ").replace("-", " ")
            console.print(f"[yellow]No prompt found, using folder name: '{prompt}'[/yellow]")
    
    try:
        console.print(f"\n[bold cyan]═══ 3D Model Quality Analysis ═══[/bold cyan]")
        console.print(f"[dim]Prompt: {prompt}[/dim]")
        console.print(f"[dim]Asset: {asset_folder}[/dim]\n")
        
        # Step 1: Extract frames - save to asset folder for inspection
        frames_dir = asset_folder / "suggestion_frames"
        console.print(f"[blue]📹 Extracting {n_frames} frames from video...[/blue]")
        console.print(f"[dim]   Saving to: {frames_dir}[/dim]")
        
        frame_paths = _extract_key_frames(
            video_path, 
            n_frames=n_frames,
            output_dir=frames_dir,
            prefix="vqa_frame"
        )
        
        if not frame_paths:
            console.print("[red]No frames extracted[/red]")
            return None
        
        console.print(f"[green]✓ Extracted {len(frame_paths)} frames to {frames_dir}[/green]")
        
        # Step 2: Generate prompt-specific questions
        console.print(f"\n[blue]🤔 Generating prompt-specific questions...[/blue]")
        try:
            custom_questions = _generate_prompt_based_questions(
                prompt, 
                settings.openai_api_key,
                n_questions=3
            )
            for i, q in enumerate(custom_questions, 1):
                console.print(f"   [dim]Q{i}: {q}[/dim]")
        except Exception as e:
            console.print(f"[yellow]Could not generate custom questions: {e}[/yellow]")
            custom_questions = [
                f"Does this model accurately represent '{prompt}'?",
                "What key features from the prompt are missing or incorrect?",
                "How well does the model capture the intended style/mood?"
            ]
        
        # Step 3: Run comprehensive analysis
        console.print(f"\n[blue]🔍 Running GPT-5.1 comprehensive analysis (medium reasoning)...[/blue]")
        console.print(f"[dim]   (Analyzing {min(6, len(frame_paths))} key frames with high detail)[/dim]")
        
        analysis = _call_gpt4o_detailed_analysis(
            frame_paths=frame_paths,
            prompt=prompt,
            custom_questions=custom_questions,
            api_key=settings.openai_api_key,
        )
        
        # Step 4: Display results
        console.print(f"\n[bold green]═══ Analysis Complete ═══[/bold green]\n")
        
        # Print section scores
        sections = analysis.get("sections", {})
        score_table = []
        for key, section in sections.items():
            score = section.get("score")
            title = section.get("title", key)
            if score is not None:
                score_str = f"{score:.1f}/10"
                if score >= 8:
                    color = "green"
                elif score >= 5:
                    color = "yellow"
                else:
                    color = "red"
                score_table.append((title, score, color))
        
        if score_table:
            console.print("[bold]Section Scores:[/bold]")
            for title, score, color in score_table:
                bar = "█" * int(score) + "░" * (10 - int(score))
                console.print(f"  {title:<30} [{color}]{bar}[/{color}] {score:.1f}/10")
        
        # Print full analysis
        console.print(f"\n[bold]Detailed Analysis:[/bold]")
        console.print("─" * 60)
        console.print(analysis.get("raw_analysis", "No analysis available"))
        console.print("─" * 60)
        
        # Step 5: Save results with properly structured sections
        results_path = asset_folder / "text3d_suggestions.json"
        with open(results_path, "w") as f:
            # Build structured sections output
            structured_sections = {}
            for key, section in sections.items():
                section_data = {
                    "title": section.get("title", key),
                    "score": section.get("score"),
                    "key_issues": section.get("key_issues", []),
                    "suggestions": section.get("suggestions", []),
                }
                # Include questions array for prompt_specific section
                if key == "prompt_specific" and "questions" in section and section["questions"]:
                    section_data["questions"] = section["questions"]
                structured_sections[key] = section_data
            
            save_data = {
                "prompt": prompt,
                "custom_questions": custom_questions,
                "n_frames_analyzed": analysis.get("n_frames_analyzed", 0),
                "sections": structured_sections,
                "frames_dir": str(frames_dir),
                "frame_paths": [str(p) for p in frame_paths],
                "raw_analysis": analysis.get("raw_analysis", ""),
            }
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        console.print(f"\n[dim]Results saved to {results_path}[/dim]")
        console.print(f"[dim]Frames saved to {frames_dir}/ (kept for inspection)[/dim]")
        
        # Note: frames are kept in the asset folder for inspection
        # They are saved to: {asset_folder}/suggestion_frames/
        
        return analysis
        
    except ImportError as e:
        console.print(f"[red]Missing dependency[/red]: {e}")
        return None
    except Exception as e:
        console.print(f"[red]Analysis failed[/red]: {e}")
        import traceback
        traceback.print_exc()
        return None


def suggest_improvements(
    asset_folder: Path,
    prompt: Optional[str] = None,
    settings: Optional[Settings] = None,
) -> Optional[Dict[str, Any]]:
    """
    Convenience alias for compute_text3d_suggestions.
    
    Run this to get detailed improvement suggestions for a 3D model.
    """
    return compute_text3d_suggestions(asset_folder, prompt, settings)


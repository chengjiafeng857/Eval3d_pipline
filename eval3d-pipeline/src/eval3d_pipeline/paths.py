from __future__ import annotations

from pathlib import Path

from .config import get_settings


def get_project_root() -> Path:
    return get_settings().project_root


def get_vendor_eval3d_root() -> Path:
    return get_settings().vendor_eval3d_root


def get_geometric_consistency_dir() -> Path:
    return get_vendor_eval3d_root() / "Eval3D" / "geometric_consistency"


def get_semantic_consistency_dir() -> Path:
    return get_vendor_eval3d_root() / "Eval3D" / "semantic_consistency"


def get_structural_consistency_dir() -> Path:
    return get_vendor_eval3d_root() / "Eval3D" / "structural_consistency"


def get_aesthetics_metric_dir() -> Path:
    return get_vendor_eval3d_root() / "Eval3D" / "aesthetics"


def get_text3d_metric_dir() -> Path:
    return get_vendor_eval3d_root() / "Eval3D" / "text_3D_alignment"


from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


def _default_project_root() -> Path:
    """Return the repository root inferred from this file location."""
    return Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_nested_delimiter="__",
        extra="ignore",
        env_prefix="EVAL3D_",
    )

    project_root: Path = Field(default_factory=_default_project_root)
    vendor_eval3d_root: Path = Field(default_factory=lambda: _default_project_root() / "vendor" / "eval3d")
    data_path: Path = Field(default_factory=lambda: _default_project_root() / "data")
    default_algorithm_name: str = "my_algo"
    gpu_ids: str = "0"
    num_gpus: int = 1
    openai_api_key: Optional[str] = Field(default=None, validation_alias="OPENAI_API_KEY")
    log_level: str = "INFO"
    default_metrics: List[str] = Field(
        default_factory=lambda: ["geometric", "semantic", "structural", "aesthetics", "text3d"]
    )

    @field_validator("project_root", "vendor_eval3d_root", "data_path")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        return value.expanduser().resolve()

    @field_validator("default_metrics", mode="before")
    @classmethod
    def _split_metrics(cls, value: Sequence[str] | str) -> List[str]:
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return list(value)

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.upper()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


"""Central configuration (env / .env). No secrets are required for the local baseline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    ollama_base_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3:8b"
    judge_model: str = "qwen3:8b"
    embed_model: str = "nomic-embed-text"
    data_dir: Path = ROOT / "data"
    skills_dir: Path = ROOT / "skills"

    max_investigation_rounds: int = 3
    max_verify_retries: int = 2
    max_tool_steps: int = 5
    trust_confidence_threshold: float = 0.7
    retrieval_k: int = 8
    llm_timeout_s: int = 180

    @property
    def app_db(self) -> Path:
        return self.data_dir / "app.db"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.db"

    @property
    def chroma_dir(self) -> Path:
        return self.data_dir / "chroma"


@lru_cache
def get_settings() -> Settings:
    s = Settings()
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s


def reset_settings_cache() -> None:
    get_settings.cache_clear()

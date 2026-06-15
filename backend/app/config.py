from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./.mltrace/mltrace.db"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    # Maximum training runs executed in parallel; mirrors the available GPUs.
    # Each running run is pinned to one GPU via CUDA_VISIBLE_DEVICES.
    max_concurrent_trainings: int = 4

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import List

load_dotenv()


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables / .env file."""

    APP_NAME: str = "MemoryOS API"
    APP_VERSION: str = "0.1.0"
    FRONTEND_URL: str = "http://localhost:5173"

    # Extra origins accepted (e.g. Vite auto-increments port to 5174 when 5173 is busy)
    EXTRA_ORIGINS: str = "http://localhost:5174,http://127.0.0.1:5173,http://127.0.0.1:5174"

    # Upload settings
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_EXTENSIONS: str = "pdf,txt"

    @property
    def cors_origins(self) -> List[str]:
        origins = [self.FRONTEND_URL]
        for o in self.EXTRA_ORIGINS.split(","):
            o = o.strip()
            if o and o not in origins:
                origins.append(o)
        return origins

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_SIZE_MB * 1024 * 1024

    @property
    def allowed_extensions_set(self) -> set:
        return {ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",")}

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()

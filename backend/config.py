from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Model Advisor"
    app_version: str = "0.1.0"
    debug: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    # API sources
    ollama_api_base: str = "https://ollama.com/api"
    huggingface_api_base: str = "https://huggingface.co/api"

    # Cache
    cache_ttl_seconds: int = 300  # 5 minutes

    # Model discovery defaults
    default_model_limit: int = 50
    max_model_limit: int = 200

    model_config = {"env_prefix": "MODEL_ADVISOR_"}


settings = Settings()

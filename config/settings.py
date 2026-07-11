"""Central settings loaded from environment variables."""

from pathlib import Path
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Auth
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

      # Groq
    groq_api_key: str
    groq_model: str = "openai/gpt-oss-20b"  # migrated: llama-3.1-8b-instant shuts down 2026-08-16
    llm_temperature: float = 0.2
    llm_max_tokens: int = 1024

    # Database
    database_url: str
    sql_agent_readonly_url: str
    sql_agent_max_rows: int = 500

    # Supabase Storage
    supabase_url: str
    supabase_service_key: str
    supabase_xray_bucket: str = "xray-uploads"
    supabase_gradcam_bucket: str = "gradcam-outputs"

    # Pinecone
    pinecone_api_key: str
    pinecone_index_name: str  = "medassist-knowledge"
    pinecone_namespace: str = "clinical"
    rag_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rag_top_k: int = 4
    rag_score_threshold: float = 0.3

    # Redis (cache + session memory)
    redis_url: str
    redis_cache_ttl_seconds: int = 2_592_000   # 30 days
    redis_session_ttl_seconds: int = 3_600       # 1 hour, sliding
    kb_version: str = "v1"        # bump after KB content updates to invalidate cache

    # CNN
    model_repo_id: str
    model_dir: Path = BASE_DIR / "models" / "weights"
    model_weights_path: Path = model_dir / "multilabel_model.pt"
    thresholds_path: Path = model_dir / "multilabel_thresholds.json"
    cnn_image_size: int = 224
    cnn_num_classes: int = 14

    # CLIP
    clip_model_name: str = "ViT-B/32"
    clip_prototype: Path = model_dir / "clip_prototype.json"
    clip_fallback_threshold: float = 0.6
    clip_valid_prompts: list[str] = [
        "a chest X-ray image",
        "a thoracic radiograph",
        "a medical X-ray of the lungs",
    ]
    clip_invalid_prompts: list[str]  = [
        "a photograph of a person",
        "a natural scene photo",
        "a document or text image",
    ]

    # GradCAM 
    # 7 zones: 6 pulmonary + 1 cardiac/mediastinum (Felson, 1973)
    gradcam_zones: list[str]         = [
        "RUZ", "LUZ", "RMZ", "LMZ", "CAR", "RLZ", "LLZ"
    ]
    gradcam_condition_colors: dict   = {
        "Atelectasis":        [255,   0,   0, 160],
        "Cardiomegaly":       [  0,   0, 255, 160],
        "Effusion":           [  0, 255,   0, 160],
        "Infiltration":       [255, 255,   0, 160],
        "Mass":               [255,   0, 255, 160],
        "Nodule":             [  0, 255, 255, 160],
        "Pneumonia":          [255, 128,   0, 160],
        "Pneumothorax":       [128,   0, 255, 160],
        "Consolidation":      [255,  20, 147, 160],
        "Edema":              [ 30, 144, 255, 160],
        "Emphysema":          [255, 215,   0, 160],
        "Fibrosis":           [139,  69,  19, 160],
        "Pleural_Thickening": [  0, 128, 128, 160],
        "Hernia":             [128, 128,   0, 160],
    }

    # Logging 
    log_dir: Path = BASE_DIR / "logs"
    log_level: str = "INFO"

    @field_validator("model_dir", "log_dir", mode="before")
    @classmethod
    def ensure_dir_exists(cls, v) -> Path:
        """Create directory if it does not exist."""
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
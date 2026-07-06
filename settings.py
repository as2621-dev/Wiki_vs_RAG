"""Environment-variable settings for the Wiki-vs-RAG benchmark harness.

Loads API keys and model configuration from a local ``.env`` file (never
committed) via ``pydantic-settings``. Import ``get_settings()`` anywhere a key
is needed rather than reading ``os.environ`` directly.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BenchmarkSettings(BaseSettings):
    """Typed settings sourced from ``.env`` / process environment."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ─── LLM providers ────────────────────────────────────────
    anthropic_api_key: str = Field(..., description="Anthropic API key for RAG generation + judge")
    voyage_api_key: str = Field(..., description="Voyage AI key for RAG chunk/question embeddings")
    pinecone_api_key: str = Field(..., description="Pinecone key for the RAG vector store")

    # ─── Model + retrieval configuration ──────────────────────
    rag_generation_model: str = Field(
        default="claude-sonnet-5", description="Model that answers from retrieved chunks"
    )
    judge_model: str = Field(
        default="claude-opus-4-8", description="Model that grades answers against gold"
    )
    voyage_embedding_model: str = Field(default="voyage-3", description="Voyage embedding model")
    voyage_embedding_dimension: int = Field(default=1024, description="voyage-3 output dimension")

    pinecone_index_name: str = Field(default="wiki-vs-rag", description="Pinecone index name")
    pinecone_cloud: str = Field(default="aws", description="Serverless cloud")
    pinecone_region: str = Field(default="us-east-1", description="Serverless region")

    retrieval_top_k: int = Field(default=8, description="Chunks retrieved per RAG query")
    chunk_size_chars: int = Field(default=2400, description="Character length of each RAG chunk")
    chunk_overlap_chars: int = Field(default=300, description="Character overlap between chunks")

    # ─── Movie-script corpus source ───────────────────────────
    movie_scripts_kaggle_dump_path: str = Field(
        default="",
        description="Local dir of <book_key>.txt scripts from the Kaggle 'Movie Transcripts 59K' dump (fallback source)",
    )

    # ─── Wiki (llmwiki) path ──────────────────────────────────
    llmwiki_mcp_config_path: str = Field(
        default="", description="Absolute path to the JSON from `./llmwiki mcp-config <dir>`"
    )
    claude_cli_binary: str = Field(default="claude", description="Claude Code CLI binary name/path")


@lru_cache
def get_settings() -> BenchmarkSettings:
    """Return a cached settings instance (parsed once per process)."""
    return BenchmarkSettings()  # type: ignore[call-arg]  # values come from env/.env

"""
AURA-Term Backend Settings & Security Configuration
===================================================
Centralized configuration module for application limits, CORS origins,
authentication boundary controls, and startup validations.
"""
import os
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class Settings:
    def __init__(self):
        self.environment: str = os.getenv("AURA_ENV", "development").lower()
        
        # CORS Allowed Origins
        # In production, explicit origins must be specified via CORS_ALLOWED_ORIGINS.
        raw_origins = os.getenv(
            "CORS_ALLOWED_ORIGINS",
            "http://localhost:8000,http://127.0.0.1:8000,http://localhost:3000"
        )
        self.cors_allowed_origins: List[str] = [
            origin.strip() for origin in raw_origins.split(",") if origin.strip()
        ]
        
        # If wildcard is explicitly provided, credentials must be disabled per W3C CORS spec.
        self.cors_allow_credentials: bool = "*" not in self.cors_allowed_origins

        # Operational Resource & Request Limits
        self.max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))  # 10 MB default
        self.max_csv_rows: int = int(os.getenv("MAX_CSV_ROWS", "10000"))                        # 10,000 rows default
        self.max_batch_rows: int = int(os.getenv("MAX_BATCH_ROWS", "5000"))                     # 5,000 JSON rows default

        # Authentication Boundary Control
        # When unset, runs in local demonstration mode.
        # When set, requires valid X-API-Key or Authorization Bearer header.
        self.api_key: Optional[str] = os.getenv("AURA_API_KEY")

        # Streaming upload chunk size (64 KB)
        self.upload_chunk_size: int = 64 * 1024

    def validate_on_startup(self):
        """Validates security-sensitive settings at application startup."""
        if self.environment == "production":
            if not self.api_key:
                logger.warning(
                    "[SECURITY WARNING] Running in PRODUCTION mode without AURA_API_KEY configured. "
                    "Operational API endpoints are currently unauthenticated."
                )
            if "*" in self.cors_allowed_origins:
                logger.error(
                    "[SECURITY RISK] Wildcard '*' CORS origin detected in production configuration."
                )
        else:
            if not self.api_key:
                logger.info(
                    "[INFO] AURA-Term initialized in local demo mode (AURA_API_KEY not set). "
                    "Endpoints accessible without authentication."
                )


settings = Settings()

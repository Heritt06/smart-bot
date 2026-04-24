from functools import lru_cache
import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "IPL Smart Bets Bot"
    app_env: str = "development"
    database_url: str = "sqlite:///./ipl_smart_bets.db"
    telegram_bot_token: str = ""
    telegram_webhook_secret: str = "telegram"
    telegram_webhook_url: str = ""
    default_bankroll: float = 10000.0
    min_edge: float = 0.05
    max_kelly_fraction: float = 0.25
    scheduler_enabled: bool = True
    cricsheet_ipl_url: str = "https://cricsheet.org/downloads/ipl_json.zip"
    ipl_data_zip_path: str = "data/ipl/ipl_json.zip"
    model_artifact_path: str = "artifacts/model_bundle.joblib"
    cricket_data_api_key: str = ""
    cricket_data_base_url: str = "https://api.cricapi.com/v1"
    auto_train_model_on_start: bool = True
    auto_refresh_data_on_start: bool = False
    port: int = 8000
    betfair_app_key: str = ""
    betfair_username: str = ""
    betfair_password: str = ""
    betfair_login_url: str = "https://identitysso.betfair.com/api/login"
    betfair_betting_url: str = "https://api.betfair.com/exchange/betting/json-rpc/v1"
    live_scan_interval_seconds: int = 60

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def public_base_url(self) -> str:
        if self.telegram_webhook_url:
            return self.telegram_webhook_url.rstrip("/")
        railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "").strip()
        if railway_domain:
            return f"https://{railway_domain}"
        return ""

    @property
    def ipl_data_zip(self) -> Path:
        return Path(self.ipl_data_zip_path)

    @property
    def model_artifact(self) -> Path:
        return Path(self.model_artifact_path)


@lru_cache
def get_settings() -> Settings:
    return Settings()

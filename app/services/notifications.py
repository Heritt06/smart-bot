from __future__ import annotations

import requests

from app.config import get_settings


def send_telegram_text(telegram_user_id: int, message: str) -> None:
    settings = get_settings()
    if not settings.telegram_bot_token:
        return

    requests.post(
        f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage",
        json={"chat_id": telegram_user_id, "text": message},
        timeout=15,
    )

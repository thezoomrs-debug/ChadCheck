"""
Центральная конфигурация бота.

Все секреты читаются из переменных окружения (.env).
При отсутствии критичных переменных приложение падает с понятной ошибкой
на старте, а не в рантайме.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Загружаем .env при локальной разработке.
# override=True: значения из .env имеют приоритет над системным окружением —
# так .env остаётся единым источником правды при локальном запуске.
# В продакшене (systemd/Docker) переменные задаются в окружении, а .env просто
# отсутствует, поэтому override ни на что не повлияет.
load_dotenv(override=True)


def _require(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(
            f"Переменная окружения {name} не задана. "
            f"Создайте файл .env по образцу .env.example."
        )
    return value


@dataclass(frozen=True)
class Config:
    # --- Telegram ---
    telegram_token: str = field(default_factory=lambda: _require("TELEGRAM_BOT_TOKEN"))

    # --- OpenRouter ---
    openrouter_api_key: str = field(default_factory=lambda: _require("OPENROUTER_API_KEY"))
    openrouter_model: str = field(
        default_factory=lambda: os.getenv("OPENROUTER_MODEL", "google/gemma-3-4b-it")
    )
    openrouter_timeout: float = field(
        default_factory=lambda: float(os.getenv("OPENROUTER_TIMEOUT", "60"))
    )


config = Config()

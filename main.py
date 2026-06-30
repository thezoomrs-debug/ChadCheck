"""
Точка входа: запуск Telegram-бота через long-polling.

Структура:
  main.py            ← запуск, диспетчер, логирование
  bot/
    config.py        ← конфиг из .env
    prompts.py       ← системный промпт + JSON-схема
    llm_client.py    ← интеграция с OpenRouter (openai/gpt-4o-mini)
    formatting.py    ← красивый вывод в Telegram
    handlers.py      ← роутер и обработчики
"""
from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.fsm.storage.memory import MemoryStorage

from bot.config import config
from bot.handlers import LoggingMiddleware, router


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.getLogger("aiogram.event").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)


async def _main() -> None:
    _setup_logging()
    log = logging.getLogger("lukstbot")

    bot = Bot(
        token=config.telegram_token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )
    dp = Dispatcher(storage=MemoryStorage())

    # Middleware должен сработать раньше роутеров — навешиваем на updates.
    dp.update.outer_middleware(LoggingMiddleware())
    dp.include_router(router)

    me = await bot.get_me()
    log.info(
        "Запущен бот @%s (id=%s) | OpenRouter: %s",
        me.username, me.id, config.openrouter_model,
    )

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()


def main() -> None:
    try:
        asyncio.run(_main())
    except (KeyboardInterrupt, SystemExit):
        logging.getLogger("lukstbot").info("Остановлено пользователем.")


if __name__ == "__main__":
    main()

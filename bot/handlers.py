"""
Хэндлеры Telegram-бота.

Маршруты:
- /start, /help — приветствие и инструкция
- фото (любого размера) — запуск анализа лица
- прочие сообщения — подсказка, что нужно прислать именно фото
"""
from __future__ import annotations

import logging
from typing import Any, Awaitable, Callable

from aiogram import BaseMiddleware, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    Chat,
    Message,
    TelegramObject,
    Update,
)

from .config import config
from .formatting import format_analysis
from .llm_client import (
    AnalysisBlockedError,
    AnalysisFailedError,
    AnalysisResult,
    LLMError,
    RateLimitError,
    analyze_face,
)
from .rate_limit import check_and_consume, format_retry_text, MAX_REQUESTS, refund

log = logging.getLogger(__name__)

router = Router()

# Максимальный размер фото, которое качаем (Telegram file_id largest).
# Боты Telegram ограничены 20 МБ скачивания через getFile — держим в уме.

_START_TEXT = (
    "👋 <b>Привет! Я бот анализа лица (looksmaxxing)</b>\n\n"
    "Пришли мне <b>фотографиию лица</b> (анфас, при хорошем свете), "
    "и я разберу её по критериям лицевой эстетики:\n\n"
    "👁 <b>Глаза</b> — canthal tilt, открытость века, симметрия\n"
    "🦴 <b>Кости</b> — челюсть, максилла, подбородок\n"
    "✨ <b>Детали</b> — кожа, соответствие стрижки\n"
    "🏆 <b>Класс</b> и потенциал улучшения\n\n"
    "📸 <i>Совет:</i> лицо прямо, без очков/масок, хороший свет.\n\n"
    "free - <b>10 анализов каждые 5 часов</b>.\n\n"
    "Отправь фото, чтобы начать 👇"
)

_HELP_TEXT = (
    "ℹ️ <b>Как пользоваться</b>\n\n"
    "1. Пришли <b>фотографию</b> лица (не файл, а именно фото).\n"
    "2. Дождись анализа (~3–5 секунд).\n"
    "3. Получишь подробный разбор с оценками.\n\n"
    "📷 Для лучшего результата:\n"
    "• Анфас, лицо занимает большую часть кадра\n"
    "• Ровный свет без резких теней\n"
    "• Без солнцезащитных очков\n\n"
    "⚠️ Оценка носит развлекательный характер и не является медицинской диагностикой."
)

_LOADING_TEXT = "🔍 Анализирую черты лица... 🔄 Это займёт около 3–5 секунд"


# --- Картинка: выбираем лучший размер ----------------------------------------

def _best_photo(message: Message) -> tuple[Any, str] | None:
    """
    Возвращает (photo_size_obj, mime_type) максимального разрешения.
    aiogram передаёт message.photo как список PhotoSize.
    """
    sizes = message.photo
    if not sizes:
        return None
    # Последний элемент — самый большой размер в Telegram API.
    best = max(sizes, key=lambda s: (s.width or 0) * (s.height or 0))
    return best, "image/jpeg"


# --- Скачивание байтов с ограничением размера ---------------------------------

_MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024  # 20 МБ — лимит getFile


# --- Команды ------------------------------------------------------------------

@router.message(Command("start"))
async def cmd_start(message: Message) -> None:
    await message.answer(_START_TEXT)


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    await message.answer(_HELP_TEXT)


# --- Главное: обработка фото --------------------------------------------------

@router.message(lambda m: m.photo)
async def handle_photo(message: Message) -> None:
    picked = _best_photo(message)
    if picked is None:
        await message.answer("🚫 Не удалось получить фото. Попробуй ещё раз.")
        return

    # --- Проверка лимита запросов (10 за 5 часов на пользователя) ---
    verdict = await check_and_consume(message.from_user.id)
    if not verdict.allowed:
        await message.answer(format_retry_text(verdict.retry_in_seconds))
        return

    photo_size, mime = picked

    # Шлём статус «анализирую» и держим ссылку, чтобы потом удалить.
    status = await message.reply(_LOADING_TEXT)

    try:
        # Скачиваем в память (BytesIO).
        file = await message.bot.download(photo_size.file_id)
        image_bytes = file.getvalue() if hasattr(file, "getvalue") else bytes(file.read())

        if not image_bytes:
            await _safe_edit(status, "🚫 Фото пришло пустым. Попробуй другое.")
            return

        if len(image_bytes) > _MAX_DOWNLOAD_BYTES:
            await _safe_edit(status, "🚫 Фото слишком большое. Пришли изображение до 20 МБ.")
            return

        result: AnalysisResult = await analyze_face(image_bytes, mime)
        text = format_analysis(result)

        # Удаляем статус и шлём финальное сообщение.
        await _safe_delete(status)
        # parse_mode=HTML; ограничиваем длину (Telegram = 4096 символов).
        final = text[:4090]
        if verdict.remaining > 0 and verdict.remaining <= 3:
            final += f"\n\n💡 Осталось анализов: <b>{verdict.remaining}</b>"
        await message.answer(final, parse_mode="HTML")

    except RateLimitError:
        # Сбой на стороне провайдера — не списываем с пользователя.
        await refund(message.from_user.id)
        await _safe_edit(
            status,
            "Упс) Слишком много запросов, попробуйте через 2-3 минуты 🙏"
        )
    except AnalysisBlockedError:
        await _safe_edit(
            status,
            "🛑 <b>Анализ заблокирован фильтрами безопасности.</b>\n\n"
            "Это бывает на обычных селфи — попробуй другое фото: "
            "с другим ракурсом, освещением или фоном."
        )
    except AnalysisFailedError as exc:
        log.warning("Analysis failed: %s", exc)
        await _safe_edit(
            status,
            "😕 <b>Не получилось выполнить анализ.</b>\n\n"
            "Возможно, лицо плохо видно или фото размыто. "
            "Попробуй чёткое фото анфаса при хорошем свете."
        )
    except LLMError as exc:
        # Внутренний сбой (сеть/таймаут) — возвращаем слот пользователю.
        await refund(message.from_user.id)
        log.error("LLMError: %s", exc)
        await _safe_edit(status, "⚠️ Внутренняя ошибка анализа. Попробуй позже.")
    except Exception as exc:  # noqa: BLE001 — последний рубеж
        await refund(message.from_user.id)
        log.exception("Неожиданная ошибка в handle_photo: %s", exc)
        await _safe_edit(status, "⚠️ Произошла непредвиденная ошибка. Попробуй ещё раз.")


# --- Фолбэк -------------------------------------------------------------------

@router.message()
async def fallback(message: Message) -> None:
    await message.answer(
        "📸 Чтобы получить анализ, пришли мне <b>фотографию</b> лица.\n"
        "Команда /help — подробная инструкция."
    )


# --- Вспомогательные безопасные операции -------------------------------------

async def _safe_edit(status: Message, text: str) -> None:
    """Редактируем сообщение-статус, молча игнорируя ошибки телеграма."""
    try:
        await status.edit_text(text, parse_mode="HTML")
    except Exception:  # noqa: BLE001
        pass


async def _safe_delete(message: Message) -> None:
    try:
        await message.delete()
    except Exception:  # noqa: BLE001
        pass


# --- Middleware: логирование апдейтов и единый отлов ошибок -------------------

class LoggingMiddleware(BaseMiddleware):
    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        chat: Chat | None = data.get("event_chat")
        try:
            return await handler(event, data)
        except Exception:
            log.exception("Ошибка при обработке апдейта (chat=%s)", chat)
            # Пытаемся вежливо сообщить пользователю.
            message: Message | None = None
            if isinstance(event, Message):
                message = event
            elif isinstance(event, CallbackQuery) and event.message:
                message = event.message
            if message is not None:
                try:
                    await message.answer("⚠️ Внутренняя ошибка. Попробуй ещё раз.")
                except Exception:  # noqa: BLE001
                    pass

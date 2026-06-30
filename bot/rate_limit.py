"""
Ограничение частоты запросов per-user (rate limiting).

Логика: скользящее окно. У каждого пользователя есть список временных меток
его запросов. Запрос разрешён, если за последние WINDOW секунд их было меньше
MAX_REQUESTS. При исчерпании лимита возвращается время до освобождения слота.

Хранилище — in-memory (словарь user_id -> [timestamps]). Подходит для одного
процесса; для нескольких воркеров потребовалось бы Redis, но боту это
не нужно.

Потокобезопасность: aiogram работает в одном event-loop, поэтому lock
используется на случай конкурентных await-точек внутри одной обработки.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass


# --- Параметры лимита (можно вынести в config при необходимости) ---------------

MAX_REQUESTS = 10           # максимум запросов...
WINDOW_SECONDS = 5 * 3600   # ...за 5 часов


# --- Результат проверки -------------------------------------------------------

@dataclass
class LimitVerdict:
    """Результат проверки лимита."""
    allowed: bool
    remaining: int        # сколько ещё запросов доступно (>= 0)
    retry_in_seconds: int # через сколько секунд освободится слот (0 если allowed)


# --- Хранилище ----------------------------------------------------------------

# user_id -> список Unix-timestamps разрешённых запросов в текущем окне.
_buckets: dict[int, list[float]] = {}
_lock = asyncio.Lock()


def _now() -> float:
    return time.monotonic()


async def check_and_consume(user_id: int) -> LimitVerdict:
    """
    Проверяет лимит и, если разрешено, записывает запрос (считает его).

    Возвращает LimitVerdict: при allowed=True лимит уже потреблён.
    """
    async with _lock:
        now = _now()
        cutoff = now - WINDOW_SECONDS

        # Чистим устаревшие метки текущего пользователя.
        bucket = [t for t in _buckets.get(user_id, []) if t > cutoff]

        if len(bucket) >= MAX_REQUESTS:
            # Лимит исчерпан. Ближайший слот освободится, когда самая старая
            # метка в окне «выпадет» за пределы WINDOW_SECONDS.
            oldest = bucket[0]
            retry_in = int(oldest + WINDOW_SECONDS - now) + 1  # +1 на округление вверх
            retry_in = max(retry_in, 1)
            _buckets[user_id] = bucket  # сохраняем очищенный список
            return LimitVerdict(
                allowed=False,
                remaining=0,
                retry_in_seconds=retry_in,
            )

        # Разрешаем: добавляем метку запроса.
        bucket.append(now)
        _buckets[user_id] = bucket
        return LimitVerdict(
            allowed=True,
            remaining=MAX_REQUESTS - len(bucket),
            retry_in_seconds=0,
        )


async def refund(user_id: int) -> None:
    """
    Возвращает один слот лимита пользователю.

    Используется, когда запрос не был выполнен по причине, не зависящей от
    пользователя (сбой провайдера, блокировка контентом, битое фото).
    Снимает последнюю добавленную метку, если она есть.
    """
    async with _lock:
        bucket = _buckets.get(user_id)
        if bucket:
            bucket.pop()  # последняя метка — самый свежий запрос


def format_retry_text(retry_in_seconds: int, remaining: int = 0) -> str:
    """
    Человекочитаемое сообщение о времени ожидания.

    retry_in_seconds: сколько секунд до освобождения слота.
    remaining: сколько запросов осталось (для информирования).
    """
    hours, rem = divmod(retry_in_seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        time_str = f"{hours} ч {minutes} мин"
    elif minutes > 0:
        time_str = f"{minutes} мин"
    else:
        time_str = f"{seconds} сек"

    return (
        f"⏳ <b>Лимит запросов исчерпан.</b>\n\n"
        f"Вы использовали все {MAX_REQUESTS} анализов за 5 часов.\n"
        f"Попробуйте снова через <b>{time_str}</b>."
    )

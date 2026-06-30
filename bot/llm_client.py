"""
Интеграция с OpenRouter API (OpenAI-совместимый формат).

Единственная публичная функция: analyze_face(image_bytes, mime_type) -> AnalysisResult.

Обрабатывает:
- превышение лимита запросов (429) → RateLimitError
- блокировку контентом → AnalysisBlockedError
- отсутствие/невалидный ответ модели → AnalysisFailedError
- сетевые ошибки и таймауты → AnalysisFailedError
"""
from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from openai import AsyncOpenAI

from .config import config
from .prompts import SYSTEM_PROMPT

log = logging.getLogger(__name__)


# --- Препроцессинг изображения -------------------------------------------------
# ЭТО ГЛАВНАЯ ОПТИМИЗАЦИЯ ПО ТОКЕНАМ: провайдер считает base64-данные картинки
# как обычные текстовые токены (~0.27 токена на байт). Полноразмерное селфи
# из Telegram (1-3 МБ) съедает 15-25к токенов.
# Vision-модели (gpt-4o-mini) сами масштабируют картинку к короткой стороне
# ~768px, поэтому подача 768px + JPEG q80 не снижает качество анализа, но
# урезает объём base64 на ~90%.
_MAX_IMAGE_SIDE = 768
_JPEG_QUALITY = 80


def _preprocess_image(image_bytes: bytes) -> tuple[bytes, str]:
    """
    Уменьшает изображение до _MAX_IMAGE_SIDE по длинной стороне и пережимает
    в JPEG. Возвращает (новые_байты, mime_type).

    Если Pillow недоступен или формат не открылся — отдаёт оригинал как есть,
    чтобы не ломать анализ ради экономии.
    """
    try:
        from PIL import Image
    except ImportError:
        log.debug("Pillow недоступен — изображение отправляется без сжатия.")
        return image_bytes, _guess_mime(image_bytes)

    try:
        img = Image.open(io.BytesIO(image_bytes))
        # Конвертируем в RGB: убираем альфа-канал, который не держит JPEG.
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        if max(img.size) > _MAX_IMAGE_SIDE:
            img.thumbnail((_MAX_IMAGE_SIDE, _MAX_IMAGE_SIDE))

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=_JPEG_QUALITY, optimize=True)
        compressed = buf.getvalue()

        log.info(
            "Препроцессинг: %d -> %d байт (%.0f%% сжатия), размер %dx%d",
            len(image_bytes), len(compressed),
            100 * (1 - len(compressed) / max(len(image_bytes), 1)),
            img.size[0], img.size[1],
        )
        return compressed, "image/jpeg"
    except Exception as exc:  # noqa: BLE001 — не роняем анализ из-за сжатия
        log.warning("Препроцессинг изображения не удался, отправляю оригинал: %s", exc)
        return image_bytes, _guess_mime(image_bytes)


def _guess_mime(image_bytes: bytes) -> str:
    """Грубое определение типа по сигнатуре, если Pillow не использовался."""
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"

# --- Кастомные ошибки ---------------------------------------------------------

class LLMError(Exception):
    """Базовая ошибка слоя LLM."""


class AnalysisBlockedError(LLMError):
    """Запрос заблокирован фильтрами безопасности."""


class RateLimitError(LLMError):
    """Превышен лимит запросов (429)."""


class AnalysisFailedError(LLMError):
    """Модель не вернула валидный результат (нет лица / пустой ответ / etc)."""


# --- Тип результата -----------------------------------------------------------

@dataclass
class AnalysisResult:
    face_detected: bool
    reason: str = ""
    canthal_tilt: str = "unknown"
    upper_eyelid_exposure: str = "unknown"
    eye_symmetry: int = 0
    jawline: int = 0
    maxilla: int = 0
    chin: int = 0
    skin_quality: int = 0
    hair_face_match: int = 0
    overall_score: int = 0
    tier: str = "unknown"
    potential_gain: int = 0
    tips: list[str] = field(default_factory=list)
    summary: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AnalysisResult":
        """Защищённо строит объект из словаря, который вернула модель."""
        def _int(key: str) -> int:
            try:
                return int(data.get(key, 0) or 0)
            except (TypeError, ValueError):
                return 0

        return cls(
            face_detected=bool(data.get("face_detected", False)),
            reason=str(data.get("reason", "") or ""),
            canthal_tilt=str(data.get("canthal_tilt", "unknown") or "unknown"),
            upper_eyelid_exposure=str(data.get("upper_eyelid_exposure", "unknown") or "unknown"),
            eye_symmetry=_int("eye_symmetry"),
            jawline=_int("jawline"),
            maxilla=_int("maxilla"),
            chin=_int("chin"),
            skin_quality=_int("skin_quality"),
            hair_face_match=_int("hair_face_match"),
            overall_score=_int("overall_score"),
            tier=str(data.get("tier", "unknown") or "unknown"),
            potential_gain=_int("potential_gain"),
            tips=[str(t) for t in (data.get("tips") or [])],
            summary=str(data.get("summary", "") or ""),
        )


# --- Асинхронный клиент OpenRouter (создаётся один раз) -----------------------

_client = AsyncOpenAI(
    api_key=config.openrouter_api_key,
    base_url="https://openrouter.ai/api/v1",
    timeout=float(config.openrouter_timeout),
)


# --- Repair-парсер обрезанного JSON -------------------------------------------

def _repair_truncated_json(text: str) -> dict[str, Any] | None:
    """
    Восстанавливает JSON, который модель не дописала из-за лимита output-токенов.
    """
    if not text:
        return None

    stack: list[str] = []
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}" and stack and stack[-1] == "{":
            stack.pop()
        elif ch == "]" and stack and stack[-1] == "[":
            stack.pop()

    repaired = text.rstrip()
    if in_string:
        repaired += '"'
    while repaired.endswith(","):
        repaired = repaired[:-1].rstrip()
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"

    try:
        result = json.loads(repaired)
        return result if isinstance(result, dict) else None
    except json.JSONDecodeError:
        log.warning("Не удалось восстановить обрезанный JSON: %r", repaired[:300])
        return None


def _parse_response_text(text: str) -> dict[str, Any]:
    """Парсит JSON-ответ модели с fallback на repair."""
    if not text:
        return {}

    text = text.strip()
    # Убираем обёртку ```json ... ```
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip("`").strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        log.warning("Модель вернула невалидный JSON, пытаюсь восстановить: %r", text[:300])

    repaired = _repair_truncated_json(text)
    if repaired is not None:
        log.info("Обрезанный JSON успешно восстановлен.")
        return repaired
    return {}


# --- Основной запрос к OpenRouter ---------------------------------------------


async def analyze_face(image_bytes: bytes, mime_type: str = "image/jpeg") -> AnalysisResult:
    """
    Асинхронно анализирует лицо на изображении через OpenRouter.

    Args:
        image_bytes: сырые байты картинки.
        mime_type: MIME-тип (image/jpeg, image/png, image/webp). Может быть
            переопределён после препроцессинга (всё пережимается в JPEG).

    Returns:
        AnalysisResult — структурированный анализ.

    Raises:
        RateLimitError: при HTTP 429.
        AnalysisBlockedError: при блокировке контентом.
        AnalysisFailedError: при сетевых ошибках, таймауте или пустом ответе.
    """
    # Сжимаем/уменьшаем картинку ДО base64: это даёт основную экономию токенов.
    image_bytes, mime_type = _preprocess_image(image_bytes)

    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    data_url = f"data:{mime_type};base64,{b64}"

    # Полная JSON-схема уже встроена в SYSTEM_PROMPT, поэтому в user-сообщении
    # её не дублируем — только короткая инструкция.
    user_text = "Проанализируй лицо. Ответь валидным JSON по схеме из системного промпта."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]

    try:
        response = await _client.chat.completions.create(
            model=config.openrouter_model,
            messages=messages,
            temperature=0.2,  # низкая температура — стабильные, повторяемые оценки
            max_tokens=512,  # JSON-ответ укладывается в ~200-300 токенов
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        err_str = str(exc).lower()

        # Rate limit: OpenRouter возвращает 429.
        if "429" in err_str or "rate" in err_str:
            raise RateLimitError("Превышен лимит запросов.") from exc

        # Блокировка контентом.
        if "content" in err_str and ("filter" in err_str or "blocked" in err_str):
            raise AnalysisBlockedError("Запрос заблокирован фильтрами контента.") from exc

        raise AnalysisFailedError(f"Сбой запроса к OpenRouter: {exc}") from exc

    # Достаём текст ответа.
    text = None
    try:
        choice = response.choices[0] if response.choices else None
        if choice and choice.message and choice.message.content:
            text = choice.message.content
    except (IndexError, AttributeError):
        pass

    if not text:
        raise AnalysisFailedError("Модель вернула пустой ответ.")

    # Логируем finish_reason для диагностики.
    try:
        finish = response.choices[0].finish_reason
        log.info("OpenRouter finish_reason=%s", finish)
        if finish == "length":
            log.warning("Ответ обрезан по max_tokens (finish_reason=length).")
    except (IndexError, AttributeError):
        pass

    data = _parse_response_text(text)
    if not data:
        raise AnalysisFailedError("Не удалось распарсить ответ модели.")

    return AnalysisResult.from_dict(data)

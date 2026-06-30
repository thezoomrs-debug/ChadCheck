"""
Превращает AnalysisResult в красивое сообщение для Telegram.

Используем только эмодзи и ASCII-полоски (█████░░░░░), чтобы шкалы
отображались моноширинно и предсказуемо в любом клиенте.
"""
from __future__ import annotations

from .llm_client import AnalysisResult

_BAR_FULL = "█"
_BAR_EMPTY = "░"
_BAR_LEN = 10

# Подписи для человекочитаемого вывода
_TILT_LABEL = {
    "positive": "положительный ↗",
    "neutral": "нейтральный →",
    "negative": "отрицательный ↘",
    "unknown": "не удалось определить",
}
_EYELID_LABEL = {
    "none": "нет",
    "mild": "слабо",
    "moderate": "умеренно",
    "heavy": "сильно",
    "unknown": "н/д",
}
_TIER_LABEL = {
    "sub-3": "Sub-3",
    "sub-5": "Sub-5",
    "ltn": "LTN",
    "mtn": "MTN",
    "htn": "HTN",
    "chad": "Chad",
    "true adam": "True Adam",
    "unknown": "—",
}
_TIER_EMOJI = {
    "sub-3": "⚠️",
    "sub-5": "🔻",
    "ltn": "📉",
    "mtn": "➖",
    "htn": "📈",
    "chad": "🔥",
    "true adam": "👑",
    "unknown": "❓",
}


def _bar(value: int) -> str:
    v = max(0, min(10, int(value)))
    return _BAR_FULL * v + _BAR_EMPTY * (_BAR_LEN - v)


def _line(name: str, value: int) -> str:
    return f"<b>{name}</b>  <code>{_bar(value)} {value}/10</code>"


def format_analysis(r: AnalysisResult) -> str:
    # Случай: лицо не найдено / фото непригодно
    if not r.face_detected:
        reason = r.reason or "не удалось корректно определить лицо на фото."
        return (
            "🚫 <b>Не получилось проанализировать</b>\n\n"
            f"{reason}\n\n"
            "📋 <b>Как получить точный анализ:</b>\n"
            "• Пришли <b>анфас</b> (лицо прямо), без сильных наклонов.\n"
            "• Хорошее освещение, без резких теней на лице.\n"
            "• Лицо должно занимать значительную часть кадра.\n"
            "• Без солнцезащитных очков и масок."
        )

    tier_label = _TIER_LABEL.get(r.tier, "—")
    tier_emoji = _TIER_EMOJI.get(r.tier, "❓")

    # Шапка
    parts: list[str] = [
        f"{tier_emoji} <b>АНАЛИЗ ЛИЦА</b> {tier_emoji}",
        f"<b>Общая оценка:</b> <code>{r.overall_score}/10</code>  "
        f"·  Класс: <b>{tier_label}</b>",
        "",
        "👁 <b>Зона глаз</b>",
        f"• Наклон глаз (canthal tilt): <b>{_TILT_LABEL.get(r.canthal_tilt, 'н/д')}</b>",
        f"• Открытость верхнего века: <b>{_EYELID_LABEL.get(r.upper_eyelid_exposure, 'н/д')}</b>",
        _line("Симметрия глаз", r.eye_symmetry),
        "",
        "🦴 <b>Костная структура</b>",
        _line("Челюсть (jawline)", r.jawline),
        _line("Максилла", r.maxilla),
        _line("Подбородок", r.chin),
        "",
        "✨ <b>Детали</b>",
        _line("Качество кожи", r.skin_quality),
        _line("Стрижка ↔ форма лица", r.hair_face_match),
        "",
        f"🚀 <b>Потенциал улучшения:</b> <code>+{r.potential_gain}/10</code>",
    ]

    # Советы
    if r.tips:
        parts.append("")
        parts.append("🛠 <b>Что можно улучшить:</b>")
        for i, tip in enumerate(r.tips, 1):
            parts.append(f"  {i}. {tip}")

    # Краткое резюме модели
    if r.summary:
        parts.append("")
        parts.append(f"💬 <i>{r.summary}</i>")

    parts.append("")
    parts.append("⚠️ <i>Оценка носит развлекательный характер и не является медицинской диагностикой.</i>")

    return "\n".join(parts)

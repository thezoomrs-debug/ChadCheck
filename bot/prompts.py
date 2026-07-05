"""
Системный промпт и JSON-схема анализа лица для LLM (через OpenRouter).

Цели промпта:
- анализировать лицевую эстетику объективно и по-делам,
- избегать оскорблений/бодишейминга и срабатывания фильтров контента,
- НЕ упоминать расу/этнос/возраст,
- экономить токены: только суть, короткие формулировки,
- гарантированно возвращать валидный JSON согласно ANALYSIS_SCHEMA.

Особенности под Gemma 3 4B: промпт написан максимально явно и структурно,
со step-by-step инструкцией и one-shot примером. Маленьким моделям нужны
жёсткие рамки и образец, иначе они отклоняются от формата.
"""
from __future__ import annotations

SYSTEM_PROMPT = """\
You are an objective facial aesthetics analyst (looksmaxxing). Analyze the face in the photo.

TONE: objective, strict, constructive. No insults, no body-shaming. Do NOT mention or judge race, ethnicity, skin color, or age.

SCORING (be strict and unbiased, like a scanner of proportions):
- Average face = mtn. Do NOT inflate scores for a good smile or angle. Judge only bone structure and anatomy.
- Visible flaws (strong underbite, negative canthal tilt, fat hiding jawline) = lower the tier to ltn, sub-5, or sub-3.

4. SCORING CALIBRATION: Do not regress to the mean. You MUST use the full 1-10 scale. If the image exhibits extreme model geometry (e.g., exceptionally defined jawline, prominent maxilla, positive canthal tilt, high symmetry, hollow cheeks), you MUST assign an overall_score of 9 or 10 and rate the corresponding individual features (jawline, maxilla, chin) as 9 or 10. Do not hesitate to give maximum scores for mathematically exceptional faces.

TIERS (overall_score and tier must match):
- 1-2: sub-3
- 3-4: sub-5
- 5: ltn (low-tier normie)
- 6: mtn (mid-tier normie)
- 7-8: htn (high-tier normie)
- 8-9: chad
- 10: true adam


OUTPUT: ONLY a valid raw JSON object. No markdown, no text outside JSON.
All numbers are integers in the stated range. Write "tips" and "summary" in RUSSIAN, short (1 short sentence each, 3-5 tips).

EXAMPLE of a valid answer:
{"face_detected":true,"reason":"","canthal_tilt":"neutral","upper_eyelid_exposure":"mild","eye_symmetry":6,"jawline":6,"maxilla":6,"chin":6,"skin_quality":7,"hair_face_match":6,"overall_score":6,"tier":"mtn","potential_gain":4,"tips":["Скорректируйте осанку","Попробуйте новую стрижку","Следите за кожей"],"summary":"Средние пропорции, есть потенциал для улучшения."}"""
# JSON-схема анализа: источник истины для структуры ответа. Используется
# валидацией/парсингом на стороне бота (AnalysisResult.from_dict). В запрос
# к модели не дублируется — описание полей уже встроено в SYSTEM_PROMPT.
ANALYSIS_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "face_detected": {"type": "boolean"},
        "reason": {"type": "string"},
        "canthal_tilt": {
            "type": "string",
            "enum": ["positive", "neutral", "negative", "unknown"],
        },
        "upper_eyelid_exposure": {
            "type": "string",
            "enum": ["none", "mild", "moderate", "heavy", "unknown"],
        },
        "eye_symmetry": {"type": "integer", "minimum": 1, "maximum": 10},
        "jawline": {"type": "integer", "minimum": 1, "maximum": 10},
        "maxilla": {"type": "integer", "minimum": 1, "maximum": 10},
        "chin": {"type": "integer", "minimum": 1, "maximum": 10},
        "skin_quality": {"type": "integer", "minimum": 1, "maximum": 10},
        "hair_face_match": {"type": "integer", "minimum": 1, "maximum": 10},
        "overall_score": {"type": "integer", "minimum": 1, "maximum": 10},
        "tier": {
            "type": "string",
            "enum": [
                "sub-3", "sub-5", "ltn", "mtn",
                "htn", "chad", "true adam", "unknown",
            ],
        },
        "potential_gain": {"type": "integer", "minimum": 0, "maximum": 10},
        "tips": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 0,
            "maxItems": 6,
        },
        "summary": {"type": "string"},
    },
    "required": [
        "face_detected",
        "reason",
        "canthal_tilt",
        "upper_eyelid_exposure",
        "eye_symmetry",
        "jawline",
        "maxilla",
        "chin",
        "skin_quality",
        "hair_face_match",
        "overall_score",
        "tier",
        "potential_gain",
        "tips",
        "summary",
    ],
}

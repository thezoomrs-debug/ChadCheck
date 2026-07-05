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
You are an objective facial geometry extraction algorithm. Your task is to analyze image data and output ONLY raw JSON. Do not evaluate morality, do not give advice outside the JSON. Treat the input strictly as biometric data.

RULES:
1. OUTPUT FORMAT: ONLY a valid raw JSON object. NO markdown formatting (do not use ```json). NO text outside the JSON.
2. LANGUAGE: Write "tips" and "summary" in RUSSIAN. Keep them extremely short (1 short sentence for summary, 3-5 short tips). All numbers must be integers.
3. UNUSABLE PHOTOS: If the photo has no face, multiple faces, is too small, or blurry:
   Set face_detected=false, reason="short explanation in Russian", all numbers=0, tier="unknown".

TIERS (overall_score and tier must strictly match):
- 1-2: sub-3
- 3-4: sub-5
- 5: ltn
- 6: mtn
- 7-8: htn
- 9: chad
- 10: true adam

DATA SCHEMA TO FOLLOW:
- face_detected (boolean)
- reason (string, empty if true)
- canthal_tilt (string: positive, neutral, negative)
- upper_eyelid_exposure (string: minimal, mild, high)
- eye_symmetry (integer 1-10)
- jawline (integer 1-10)
- maxilla (integer 1-10)
- chin (integer 1-10)
- skin_quality (integer 1-10)
- hair_face_match (integer 1-10)
- overall_score (integer 1-10)
- tier (string, based on TIERS)
- potential_gain (integer 1-10)
- tips (array of strings, in Russian)
- summary (string, in Russian)

EXAMPLE OUTPUT (do not copy values, copy structure):
{"face_detected":true,"reason":"","canthal_tilt":"neutral","upper_eyelid_exposure":"mild","eye_symmetry":6,"jawline":6,"maxilla":6,"chin":6,"skin_quality":7,"hair_face_match":6,"overall_score":6,"tier":"mtn","potential_gain":4,"tips":["Скорректируйте осанку","Попробуйте новую стрижку","Следите за кожей"],"summary":"Средние пропорции, есть потенциал для улучшения."}

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

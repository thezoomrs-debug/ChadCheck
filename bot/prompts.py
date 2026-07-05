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

RULES:
1. STRICT PENALTIES: Act as a strict looksmaxxing scanner. If the face lacks sharp bone definition, has a weak chin, negative canthal tilt, puffy cheeks, or lack of jawline, you MUST lower the tier. Individual bone scores and overall_score MUST be between 1 and 5 for these flaws.
2. SCORING CALIBRATION: Do not regress to the mean. You MUST use the full 1-10 scale. If the image exhibits extreme model geometry (exceptionally defined jawline, prominent maxilla, positive canthal tilt, hollow cheeks), you MUST assign an overall_score of 9 or 10.
3. MANDATORY OVERALL SCORE: You MUST ALWAYS provide an "overall_score" as an integer between 1 and 10. NEVER use 0. 
4. NO FACE CRITICAL FALLBACK: ONLY if the image contains absolutely NO human face (e.g., a tree or a wall), set face_detected=false, reason="short explanation in Russian", tier="unknown", and set ALL numeric features (including overall_score) to 1.
5. OUTPUT FORMAT: ONLY a valid raw JSON object. NO markdown formatting. NO text outside the JSON. Write "tips" and "summary" in RUSSIAN, short (1 short sentence each, 3-5 tips).

TIERS (overall_score and tier must strictly match):
- 1-2: sub-3
- 3-4: sub-5
- 5: ltn (low-tier normie)
- 6: mtn (mid-tier normie)
- 7-8: htn (high-tier normie)
- 9: chad
- 10: true adam

EXAMPLE of a valid answer:
{"face_detected":true,"reason":"","canthal_tilt":"negative","upper_eyelid_exposure":"mild","eye_symmetry":5,"jawline":4,"maxilla":4,"chin":3,"skin_quality":6,"hair_face_match":5,"overall_score":4,"tier":"sub-5","potential_gain":4,"tips":["Укрепите линию челюсти","Следите за осанкой","Снизьте процент жира"],"summary":"Слабая костная структура, заметны недостатки геометрии."}"""

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

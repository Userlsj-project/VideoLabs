"""
[팀 직접 구현] F-7: A/B 프롬프트 자동 변형 생성.

기본 프롬프트에서 조명·속도·감정 3가지 축으로 변형을 자동 생성한다.
어떤 변형이 더 좋은 피드백을 받았는지 추적해 프롬프트 최적화에 활용.
경쟁사(Runway, Pika)에는 없는 A/B 테스트 자동화 기능.
"""

from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

_DIMENSIONS = [
    {
        "key": "lighting",
        "label": "조명/색감 강조",
        "instruction": "조명, 색감, 밝기 표현을 강화하도록 변형해주세요. 예: 'warm golden hour lighting', 'cinematic color grading', 'soft diffused light' 등의 표현을 추가하세요.",
    },
    {
        "key": "pacing",
        "label": "속도/편집 리듬 강조",
        "instruction": "카메라 움직임, 장면 전환 속도, 편집 리듬 표현을 강화하도록 변형해주세요. 예: 'dynamic cuts', 'slow reveal', 'fast-paced montage' 등의 표현을 추가하세요.",
    },
    {
        "key": "emotion",
        "label": "감정/분위기 강조",
        "instruction": "감정적 분위기, 톤, 느낌 표현을 강화하도록 변형해주세요. 예: 'uplifting atmosphere', 'tense dramatic mood', 'heartwarming emotional tone' 등의 표현을 추가하세요.",
    },
]


def generate_prompt_variants(
    base_prompt: str,
    gemini_api_key: str | None,
    gemini_model: str = "gemini-2.0-flash-lite",
    n_variants: int = 3,
) -> list[dict]:
    """
    기본 프롬프트에서 각기 다른 스타일의 변형 3개를 생성한다.

    Returns:
        [{"key": "lighting", "label": "조명/색감 강조", "prompt": "...변형된 프롬프트..."}]
    """
    if not base_prompt:
        return []

    dimensions = _DIMENSIONS[:n_variants]

    if not gemini_api_key:
        # API 키 없으면 규칙 기반 간단 변형
        return _rule_based_variants(base_prompt, dimensions)

    variants = []
    try:
        from google import genai
        client = genai.Client(api_key=gemini_api_key)

        for dim in dimensions:
            prompt = (
                f"다음 영상 생성 프롬프트를 '{dim['label']}' 측면을 강조하도록 변형해주세요.\n"
                f"원본 프롬프트의 핵심 의도와 주제는 절대 바꾸지 말고, {dim['instruction']}\n"
                f"변형된 프롬프트만 출력하세요 (설명·주석 없이).\n\n"
                f"원본: {base_prompt[:800]}"
            )
            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
            )
            variant_text = (response.text or "").strip()
            if variant_text:
                variants.append({
                    "key": dim["key"],
                    "label": dim["label"],
                    "prompt": variant_text,
                })

    except Exception as e:
        logger.warning("[PromptVariants] Gemini API 실패, 규칙 기반으로 폴백: %s", e)
        return _rule_based_variants(base_prompt, dimensions)

    return variants


def _rule_based_variants(base_prompt: str, dimensions: list[dict]) -> list[dict]:
    """API 없을 때 규칙 기반 변형 생성."""
    additions = {
        "lighting": "Warm cinematic color grading, soft golden hour lighting, rich contrast.",
        "pacing": "Dynamic fast cuts, smooth transitions, energetic editing rhythm.",
        "emotion": "Uplifting emotional atmosphere, heartwarming tone, engaging mood.",
    }
    variants = []
    for dim in dimensions:
        add = additions.get(dim["key"], "")
        variants.append({
            "key": dim["key"],
            "label": dim["label"],
            "prompt": f"{base_prompt.rstrip('.')}. {add}",
        })
    return variants

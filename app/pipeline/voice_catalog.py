"""
[팀 직접 설계] ElevenLabs 보이스 카탈로그.

각 보이스의 특성을 태깅해두고, 피드백에 따라 자동으로 최적 보이스를 선택한다.

voice_id 추가 방법:
  1. ElevenLabs 라이브러리(elevenlabs.io/voice-library)에서 한국어 보이스 미리듣기
  2. 보이스 ID 복사
  3. 아래 VOICE_CATALOG에 항목 추가
  4. is_placeholder=False 로 변경
"""

from __future__ import annotations

# ── 보이스 카탈로그 ────────────────────────────────────────────────────────────
# tags: 피드백 매칭에 사용되는 특성 태그
# weakness: 이 보이스가 취약한 피드백 키워드 (이 보이스를 피해야 할 상황)
# best_for: 어떤 영상에 어울리는지 (UI 설명용)

VOICE_CATALOG: list[dict] = [
    {
        "voice_id": "WnFTM7Gw75NWybKycaRt",
        "name": "Victor - 남성 드라마틱",
        "gender": "male",
        "style": "dramatic",
        "language": "ko",
        "tags": ["남성", "드라마틱", "중후한", "권위있는", "홍보"],
        "weakness": ["느끼함", "AI티", "과장됨", "어색함"],
        "best_for": "드라마틱한 홍보영상, 다큐멘터리, 시네마틱 내레이션",
        "is_placeholder": False,
    },
    {
        "voice_id": "s07IwTCOrCDCaETjUVjx",
        "name": "Hyunbin - 남성 아나운서",
        "gender": "male",
        "style": "formal",
        "language": "ko",
        "tags": ["남성", "공식적", "아나운서", "절제된", "신뢰감", "기업", "자연스러운"],
        "weakness": ["단조로움", "차가움"],
        "best_for": "기업/학교 공식 홍보, 뉴스형 영상, 기업 PR",
        "is_placeholder": False,
    },
    {
        "voice_id": "HvCQYI15cbdFVC9pSBfb",
        "name": "Mr.J - 남성 캐주얼",
        "gender": "male",
        "style": "casual",
        "language": "ko",
        "tags": ["남성", "캐주얼", "유튜브", "친근한", "설명형"],
        "weakness": ["공식 홍보", "격식", "기관"],
        "best_for": "유튜브 콘텐츠, 제품 리뷰, 튜토리얼 (공식 홍보영상 비추천)",
        "is_placeholder": False,
    },
    {
        "voice_id": "ZJCNdZEjYwkOElxugmW2",
        "name": "Hyuk - 남성 감성적",
        "gender": "male",
        "style": "emotional",
        "language": "ko",
        "tags": ["남성", "감성적", "중후한", "스토리텔링", "따뜻한"],
        "weakness": ["과장됨"],
        "best_for": "감성 스토리텔링, 드라마틱 내레이션, 감동적인 영상",
        "is_placeholder": False,
    },
    {
        "voice_id": "iWLjl1zCuqXRkW6494ve",
        "name": "Jisoo - 여성 아나운서",
        "gender": "female",
        "style": "formal",
        "language": "ko",
        "tags": ["여성", "공식적", "아나운서", "명확한", "신뢰감", "뉴스"],
        "weakness": ["차가움", "단조로움"],
        "best_for": "공식 발표, 학술·기업 홍보, 뉴스형 영상",
        "is_placeholder": False,
    },
    {
        "voice_id": "FEMALE_NATURAL_ID_HERE",
        "name": "여성 자연스러운",
        "gender": "female",
        "style": "natural",
        "language": "ko",
        "tags": ["여성", "자연스러운", "친근한", "따뜻한"],
        "weakness": [],
        "best_for": "친근한 소개 영상, 라이프스타일 홍보",
        "is_placeholder": True,
    },
    {
        "voice_id": "FEMALE_ENERGETIC_ID_HERE",
        "name": "여성 활기찬",
        "gender": "female",
        "style": "energetic",
        "language": "ko",
        "tags": ["여성", "활기찬", "밝은", "젊은", "에너지"],
        "weakness": ["과장됨"],
        "best_for": "젊은 타겟 홍보, 이벤트·축제 영상",
        "is_placeholder": True,
    },
]


def get_all_voices() -> list[dict]:
    return VOICE_CATALOG


def get_active_voices() -> list[dict]:
    """플레이스홀더가 아닌 실제 설정된 보이스만 반환."""
    return [v for v in VOICE_CATALOG if not v.get("is_placeholder")]


def get_voice_by_id(voice_id: str) -> dict | None:
    for v in VOICE_CATALOG:
        if v["voice_id"] == voice_id:
            return v
    return None


def register_voice(
    voice_id: str,
    name: str,
    gender: str,
    style: str,
    tags: list[str],
    weakness: list[str],
    best_for: str,
    language: str = "ko",
) -> None:
    """런타임에 새 보이스를 카탈로그에 추가 (플레이스홀더 대체)."""
    for v in VOICE_CATALOG:
        if v.get("is_placeholder") and v["gender"] == gender and v["style"] == style:
            v.update({
                "voice_id": voice_id,
                "name": name,
                "tags": tags,
                "weakness": weakness,
                "best_for": best_for,
                "language": language,
                "is_placeholder": False,
            })
            return
    VOICE_CATALOG.append({
        "voice_id": voice_id,
        "name": name,
        "gender": gender,
        "style": style,
        "language": language,
        "tags": tags,
        "weakness": weakness,
        "best_for": best_for,
        "is_placeholder": False,
    })

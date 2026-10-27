"""
[팀 직접 구현] F-8: 자동 댓글 답글 초안 생성.

승인된 피드백 클러스터에 대해 브랜드 일관된 답글 초안을 생성한다.
크리에이터가 검토 후 YouTube API로 게시할 수 있다.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def generate_reply_drafts(
    clusters: list,
    genre: str,
    gemini_api_key: str | None,
    gemini_model: str = "gemini-2.0-flash-lite",
) -> list[dict]:
    """
    승인된 클러스터에 대한 댓글 답글 초안을 생성한다.

    Args:
        clusters: ClusterResult 객체 목록
        genre: 영상 장르 (브랜드 톤 결정에 사용)
        gemini_api_key: Gemini API 키
        gemini_model: Gemini 모델명

    Returns:
        [{"cluster_label": "...", "cluster_key": "...", "draft": "..."}]
    """
    approved = [c for c in clusters if c.status == "approved"]
    if not approved:
        return []

    if not gemini_api_key:
        return _rule_based_replies(approved, genre)

    replies = []
    try:
        from google import genai
        client = genai.Client(api_key=gemini_api_key)

        for cluster in approved:
            label = cluster.label or "(피드백 주제)"
            summary = cluster.cluster_summary or cluster.representative_text or ""
            category = cluster.category or "general"

            prompt = (
                f"한국어 영상 크리에이터로서 시청자 피드백에 답글을 작성해주세요.\n\n"
                f"영상 장르: {genre}\n"
                f"피드백 주제: {label}\n"
                f"피드백 요약: {summary[:200]}\n"
                f"피드백 카테고리: {category}\n\n"
                f"규칙:\n"
                f"- 1-2문장으로 간결하게\n"
                f"- 감사 표현 포함\n"
                f"- 다음 영상에서 반영 예정임을 구체적으로 언급\n"
                f"- 따뜻하고 진정성 있는 톤\n"
                f"- 한국어로 작성\n\n"
                f"답글만 출력하세요 (설명 없이)."
            )

            response = client.models.generate_content(
                model=gemini_model,
                contents=prompt,
            )
            draft = (response.text or "").strip()
            if draft:
                replies.append({
                    "cluster_label": label,
                    "cluster_key": cluster.cluster_key,
                    "draft": draft,
                    "category": category,
                })

    except Exception as e:
        logger.warning("[ReplyGenerator] Gemini API 실패, 규칙 기반으로 폴백: %s", e)
        return _rule_based_replies(approved, genre)

    return replies


def _rule_based_replies(clusters: list, genre: str) -> list[dict]:
    """API 없을 때 템플릿 기반 답글 생성."""
    templates = {
        "audio": "소중한 오디오 관련 피드백 감사합니다! 다음 영상에서 배경음악 볼륨과 음질을 개선하겠습니다. 😊",
        "visual": "화질과 시각적 구성에 대한 피드백 감사합니다! 다음 영상에서 조명과 색감을 더 신경 쓰겠습니다.",
        "subtitle": "자막 관련 의견 주셔서 감사합니다! 다음 영상에서 자막 크기와 타이밍을 개선할게요.",
        "pacing": "편집 속도에 대한 의견 감사합니다! 다음 영상에서 장면 전환을 더 자연스럽게 만들겠습니다.",
        "story": "스토리 관련 피드백 감사합니다! 다음 영상에서 내용 구성을 더 탄탄하게 하겠습니다.",
        "content": "내용에 대한 소중한 피드백 감사합니다! 다음 영상에서 더 풍부한 정보를 담겠습니다.",
    }
    default = "소중한 피드백 감사합니다! 다음 영상에서 반영하도록 노력하겠습니다. 항상 응원해 주셔서 감사해요! 😊"

    replies = []
    for cluster in clusters:
        category = cluster.category or "other"
        draft = templates.get(category, default)
        replies.append({
            "cluster_label": cluster.label or "(주제)",
            "cluster_key": cluster.cluster_key,
            "draft": draft,
            "category": category,
        })
    return replies

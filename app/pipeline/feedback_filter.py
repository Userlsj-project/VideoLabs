"""
Step 6: 피드백 필터링
댓글을 constructive / non_constructive / spam / inappropriate로 분류한다.
"건설적 피드백" 기준은 어조(긍정/부정)가 아니라 구체적 개선 제안 포함 여부다.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime


@dataclass
class FeedbackClassification:
    comment_text: str
    category: str  # constructive | non_constructive | spam | inappropriate
    improvement_suggestion: str | None
    confidence: float
    reasoning: str


def classify_comments(
    comments: list[str],
    api_key: str | None = None,
    model: str = "claude-sonnet-4-6",
) -> list[FeedbackClassification]:
    """
    댓글 목록을 분류한다.
    api_key가 없으면 규칙 기반 폴백을 사용한다.
    """
    if not comments:
        return []

    if api_key:
        try:
            return _classify_with_claude(comments, api_key, model)
        except Exception as e:
            print(f"[FeedbackFilter] Claude 분류 실패, 규칙 기반으로 폴백: {e}")

    return _classify_with_rules(comments)


def _classify_with_claude(
    comments: list[str], api_key: str, model: str
) -> list[FeedbackClassification]:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    BATCH = 20
    results: list[FeedbackClassification] = []

    for start in range(0, len(comments), BATCH):
        batch = comments[start : start + BATCH]
        numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(batch))

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system="""당신은 유튜브 댓글 분류 전문가입니다. 항상 JSON만 응답합니다.""",
            messages=[
                {
                    "role": "user",
                    "content": f"""다음 유튜브 댓글들을 분류해주세요.

분류 기준:
- constructive: 어조(긍정/부정)와 무관하게 구체적인 개선 제안이 포함된 건설적 피드백
  예: "음악이 너무 크다, 줄여주세요" → constructive
  예: "자막이 안 보여요, 색 바꿔주세요" → constructive (부정적 어조여도 OK)
- non_constructive: 개선 제안 없는 단순 감상평·칭찬·일반 반응
- spam: 스팸·광고·무관한 내용·다른 채널 홍보
- inappropriate: 저작권 침해 요구·성적/폭력적 콘텐츠 요구·혐오 표현

핵심: 부정적 어조 ≠ non_constructive. 구체적 개선 제안이 있으면 constructive입니다.

댓글:
{numbered}

JSON 응답:
{{
  "results": [
    {{
      "index": 1,
      "category": "constructive",
      "improvement_suggestion": "개선 제안 요약 (constructive만)",
      "confidence": 0.95,
      "reasoning": "분류 근거"
    }}
  ]
}}""",
                }
            ],
        )

        text = response.content[0].text.strip()
        json_match = re.search(r"\{.*\}", text, re.DOTALL)
        if not json_match:
            results.extend(_classify_with_rules(batch))
            continue

        data = json.loads(json_match.group())
        for item in data.get("results", []):
            idx = item["index"] - 1
            if idx < len(batch):
                results.append(
                    FeedbackClassification(
                        comment_text=batch[idx],
                        category=item.get("category", "non_constructive"),
                        improvement_suggestion=item.get("improvement_suggestion"),
                        confidence=float(item.get("confidence", 0.8)),
                        reasoning=item.get("reasoning", ""),
                    )
                )

    return results


def _classify_with_rules(comments: list[str]) -> list[FeedbackClassification]:
    """규칙 기반 폴백 분류기 (한국어 + 영어 지원)."""
    constructive_keywords = [
        # 한국어
        "줄여", "키워", "늘려", "바꿔", "넣어", "추가", "수정", "개선",
        "느리게", "빠르게", "크게", "작게", "밝게", "어둡게",
        "해주세요", "주세요", "좋을 것 같", "어떨까요", "어떨까",
        "힘들어요", "안 들려", "안 보여", "읽기 어렵", "이해하기 어렵",
        "아쉽", "부족", "짧아", "길어", "단조롭",
        # 영어
        "please", "should", "could", "would be better", "too loud", "too small",
        "too large", "too long", "too short", "too fast", "too slow",
        "lower", "higher", "increase", "decrease", "improve", "fix",
        "cannot hear", "can't hear", "hard to read", "difficult to",
        "suggest", "recommend", "change", "add", "remove",
    ]
    spam_keywords = [
        # 한국어
        "구독", "좋아요 눌러", "방문해주세요", "채널", "팔로우", "맞구독",
        # 영어
        "subscribe", "follow me", "check out my", "visit my", "click here",
        "http://", "https://", ".com", ".kr", "click",
    ]
    inappropriate_keywords = [
        "저작권", "표절", "불법", "야한", "성인", "19금", "폭력", "혐오",
        "copyright", "plagiarism", "illegal", "nsfw", "explicit",
    ]

    results = []
    for comment in comments:
        text_lower = comment.lower()

        if any(kw in text_lower for kw in inappropriate_keywords):
            results.append(
                FeedbackClassification(
                    comment_text=comment,
                    category="inappropriate",
                    improvement_suggestion=None,
                    confidence=0.75,
                    reasoning="부적절한 키워드 포함",
                )
            )
        elif any(kw in text_lower for kw in spam_keywords):
            results.append(
                FeedbackClassification(
                    comment_text=comment,
                    category="spam",
                    improvement_suggestion=None,
                    confidence=0.75,
                    reasoning="스팸 키워드 포함",
                )
            )
        elif any(kw in comment for kw in constructive_keywords):
            results.append(
                FeedbackClassification(
                    comment_text=comment,
                    category="constructive",
                    improvement_suggestion=comment,
                    confidence=0.65,
                    reasoning="개선 관련 키워드 포함",
                )
            )
        else:
            results.append(
                FeedbackClassification(
                    comment_text=comment,
                    category="non_constructive",
                    improvement_suggestion=None,
                    confidence=0.60,
                    reasoning="개선 제안 없는 일반 댓글",
                )
            )

    return results

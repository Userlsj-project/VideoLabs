"""
[팀 직접 구현] 피드백(댓글)을 특정 샷에 매핑.

우선순위:
  1. 댓글에 타임스탬프(0:15 등) 있음 → timestamp_parser로 샷 특정
  2. 타임스탬프 없음 + Gemini API 키 있음 → Gemini가 어떤 샷인지 추론
  3. 매핑 불가 → shot_round_id=None (전체 영상 피드백으로 처리)

하위 호환성: ShotRound 레코드가 없으면 조용히 건너뜀.
"""

from __future__ import annotations
import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..models import Feedback, ShotRound

from .utils.timestamp_parser import extract_timestamp, map_timestamp_to_shot_number

logger = logging.getLogger(__name__)


def map_feedbacks_to_shots(
    feedbacks: list,       # list[Feedback]
    shots: list,           # list[ShotRound]
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-2.0-flash-lite",
) -> None:
    """
    각 피드백의 shot_round_id / timestamp_seconds를 제자리(in-place) 업데이트.

    Args:
        feedbacks: 분류가 완료된 Feedback 객체 목록 (constructive만 전달 권장)
        shots: 현재 라운드의 ShotRound 목록
        gemini_api_key: 없으면 타임스탬프 기반 매핑만 수행
        gemini_model: Gemini 텍스트 모델명
    """
    if not shots:
        return  # ShotRound 없음 = 단일 생성 모드 → 건너뜀

    shot_map: dict[int, object] = {s.shot_number: s for s in shots}

    # 1단계: 타임스탬프 기반 매핑
    unresolved: list = []
    for fb in feedbacks:
        ts = extract_timestamp(fb.comment_text or "")
        if ts is not None:
            fb.timestamp_seconds = ts
            shot_num = map_timestamp_to_shot_number(ts, shots)
            if shot_num and shot_num in shot_map:
                fb.shot_round_id = shot_map[shot_num].id
                logger.debug("[ShotMapper] 타임스탬프 %.1f초 → 샷 %d", ts, shot_num)
            else:
                unresolved.append(fb)
        else:
            unresolved.append(fb)

    if not unresolved or not gemini_api_key:
        if unresolved:
            logger.info("[ShotMapper] Gemini 키 없음 — %d개 미매핑 피드백은 전체 영상으로 처리", len(unresolved))
        return

    # 2단계: Gemini 기반 추론 매핑
    _map_with_gemini(unresolved, shots, gemini_api_key, gemini_model, shot_map)


def _map_with_gemini(
    feedbacks: list,
    shots: list,
    api_key: str,
    model: str,
    shot_map: dict,
) -> None:
    """Gemini에 댓글 묶음을 보내 샷 번호를 추론."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel(model)
    except Exception as e:
        logger.warning("[ShotMapper] Gemini 초기화 실패: %s", e)
        return

    # 샷 요약 구성
    shot_summaries = "\n".join(
        f"샷{s.shot_number} ({s.start_time_seconds:.0f}초~{s.end_time_seconds:.0f}초): "
        f"{(s.prompt or '')[:120]}"
        for s in sorted(shots, key=lambda x: x.shot_number)
    )

    # 댓글 인덱스 → 피드백 매핑
    indexed = {i: fb for i, fb in enumerate(feedbacks)}

    # 배치 처리 (최대 20개씩)
    batch_size = 20
    items = list(indexed.items())
    for batch_start in range(0, len(items), batch_size):
        batch = items[batch_start: batch_start + batch_size]
        comments_block = "\n".join(
            f"[{i}] {fb.comment_text[:200]}"
            for i, fb in batch
        )
        prompt = (
            "다음은 영상의 샷 구성입니다:\n"
            f"{shot_summaries}\n\n"
            "아래 시청자 댓글 각각이 어떤 샷에 관한 내용인지 번호로 답하세요.\n"
            "해당 샷을 특정할 수 없으면 null로 답하세요.\n"
            "JSON 배열로만 응답: [{\"idx\": 0, \"shot\": 2}, {\"idx\": 1, \"shot\": null}, ...]\n\n"
            "<viewer_feedback>\n"
            f"{comments_block}\n"
            "</viewer_feedback>"
        )

        try:
            resp = client.generate_content(prompt)
            raw = resp.text.strip()
            # JSON 추출 (```json ... ``` 블록 처리)
            if "```" in raw:
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            result = json.loads(raw)
            for entry in result:
                idx = entry.get("idx")
                shot_num = entry.get("shot")
                if idx is not None and shot_num and shot_num in shot_map:
                    fb = indexed.get(idx)
                    if fb:
                        fb.shot_round_id = shot_map[shot_num].id
                        logger.debug("[ShotMapper] Gemini 추론 → 댓글 %d = 샷 %d", idx, shot_num)
        except Exception as e:
            logger.warning("[ShotMapper] Gemini 배치 매핑 실패: %s", e)


def compute_shot_distribution(feedbacks: list) -> dict[int | None, int]:
    """
    shot_round_id별 피드백 수 집계.
    None = 전체 영상 피드백.

    Returns: {shot_round_id: count}
    """
    dist: dict[int | None, int] = {}
    for fb in feedbacks:
        key = fb.shot_round_id
        dist[key] = dist.get(key, 0) + 1
    return dist


def get_shots_needing_regeneration(
    approved_cluster_keys: list[str],
    shots: list,
) -> list[int]:
    """
    승인된 클러스터의 target_shot_number를 수집해 재생성이 필요한 샷 번호 목록 반환.
    None target = 전체 영상 영향 → 모든 샷 재생성.

    Returns: 재생성 필요 샷 번호 목록 (정렬됨), 비어 있으면 전체 영상 재생성
    """
    from ..models import ClusterResult
    affected: set[int] = set()
    needs_full_regen = False

    for key in approved_cluster_keys:
        cr = ClusterResult.query.filter_by(cluster_key=key).first()
        if cr is None:
            continue
        if cr.target_shot_number is None:
            needs_full_regen = True
            break
        affected.add(cr.target_shot_number)

    if needs_full_regen:
        return []  # 빈 리스트 = 전체 재생성 신호

    return sorted(affected)

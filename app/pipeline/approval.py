"""
[팀 직접 설계·구현] 클러스터 승인 워크플로 + 프롬프트 버전 관리.

역할:
  - 클러스터링 결과(RawCluster)를 DB(ClusterResult)에 저장
  - 사용자가 승인/거절/보류한 클러스터를 프롬프트에 반영
  - 반영 전 unified diff 미리보기 제공
  - PromptVersion 레코드 생성 (버전 번호 단조 증가)
  - 이전 버전으로 롤백
  - 재발(reoccurrence) 감지: 동일 라벨이 이전 라운드에도 있었는지 확인
  - GlobalTechnicalSetting 승격: scope='technical' 클러스터 전역 반영
"""

from __future__ import annotations

import difflib
import json
import logging
import re
from datetime import datetime

from ..models import (
    ClusterResult,
    Feedback,
    GlobalTechnicalSetting,
    PromptVersion,
    VideoRound,
    db,
)
from .clusterer import RawCluster

logger = logging.getLogger(__name__)

# ── 간접 프롬프트 인젝션 감지 패턴 ──────────────────────────────────────────
# 유튜브 댓글(외부 입력)에 지시문이 숨겨진 경우를 감지해 needs_review=True 설정
_INJECTION_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"무시\s*(해|하고|하세요|해라)",
        r"이전\s*(지시|명령|프롬프트).*무시",
        r"위\s*(의|에\s*있는|쪽)?\s*(지시|명령|내용).*무시",
        r"(시스템|이전|숨겨진|새로운)\s*지시문",  # "지시문" 단독 제거 — 과탐지 방지
        r"ignore\s*(previous|all|above|the\s+)?instructions?",
        r"disregard\s*(previous|all|above|the\s+)?instructions?",
        r"forget\s*(previous|all|the\s+)?instructions?",
        r"override\s*(previous|all|above|the\s+)?instructions?",
        r"new\s+(instruction|directive|role|persona)",
        r"system\s*prompt",
        r"<\s*/?system\s*>",
        r"###\s*system",
        r"명령\s*실행",
        r"act\s*as\s*(an?\s*)?(ai|assistant|gpt|claude)",
        r"you\s*are\s*(now|a)\s*(an?\s*)?(ai|assistant|different)",
        r"jailbreak",
        r"프롬프트\s*(변경|수정|삽입|주입)",
        r"prompt\s*injection",
    ]
]


def _detect_injection(text: str) -> bool:
    """피드백 텍스트에 인젝션 의심 패턴이 있으면 True 반환."""
    return any(p.search(text) for p in _INJECTION_PATTERNS)

# ── 클러스터 저장 ────────────────────────────────────────────────────────────


def save_clusters(
    raw_clusters: list[RawCluster],
    video_round: VideoRound,
    session_feedbacks: list[Feedback],
) -> list[ClusterResult]:
    """
    [팀 직접 구현] RawCluster 목록을 DB ClusterResult로 저장.

    - Feedback.cluster_key, cluster_label 업데이트
    - 재발 횟수(reoccurrence_count) 계산
    - 기존 pending 레코드가 있으면 덮어쓰지 않고 업데이트

    Args:
        raw_clusters: clusterer.run_clustering()의 반환값
        video_round: 현재 라운드 ORM 객체
        session_feedbacks: 이 라운드의 모든 Feedback 객체 (인덱스 순서 일치)

    Returns:
        저장된 ClusterResult 목록
    """
    # 피드백 인덱스 → Feedback 객체 매핑 (raw_clusters가 사용하는 인덱스 기준)
    idx_to_feedback: dict[int, Feedback] = {
        i: f for i, f in enumerate(session_feedbacks)
    }

    saved: list[ClusterResult] = []

    for raw in raw_clusters:
        # 재발 감지: 같은 세션의 이전 라운드에서 동일 label이 있었는지
        prev_count = _count_reoccurrences(
            label=raw.label,
            category=raw.category,
            current_round_id=video_round.id,
            session_id=video_round.session_id,
        )

        # 재발 이슈는 우선순위를 라운드당 10%씩 최대 30% 상향 조정
        base_priority = raw.__dict__.get("priority_score", 0.0)
        if prev_count > 0:
            reoccurrence_boost = 1.0 + 0.10 * min(prev_count, 3)
            base_priority = min(base_priority * reoccurrence_boost, 1.0)

        cr = ClusterResult(
            video_round_id=video_round.id,
            cluster_key=raw.cluster_key,
            category=raw.category,
            scope=raw.scope,
            label=raw.label,
            representative_text=raw.representative_text,
            count=raw.count,
            avg_similarity=raw.avg_similarity,
            priority_score=round(base_priority, 4),
            freq_score=raw.__dict__.get("freq_score", 0.0),
            avg_validity_score=raw.__dict__.get("avg_validity_score", 0.5),
            avg_severity_score=raw.avg_severity_score,
            avg_urgency_score=raw.avg_urgency_score,
            avg_quality_score=raw.avg_quality_score,
            cohesion_score=raw.cohesion_score,
            cluster_summary=raw.cluster_summary,
            contradicted_by_key=raw.contradicted_by_key,
            status="pending",
            reoccurrence_count=prev_count,
            first_seen_round=(
                video_round.round_number if prev_count == 0
                else _first_seen_round(raw.label, raw.category, video_round.session_id)
            ),
        )
        db.session.add(cr)

        # 피드백에 cluster_key 역참조 저장
        for orig_idx in raw.feedback_indices:
            fb = idx_to_feedback.get(orig_idx)
            if fb:
                fb.cluster_key = raw.cluster_key
                fb.cluster_label = raw.label

        saved.append(cr)

    db.session.flush()  # id 할당
    return saved


def _count_reoccurrences(
    label: str,
    category: str,
    current_round_id: int,
    session_id: int,
) -> int:
    """같은 세션의 이전 라운드에서 동일 label+category 클러스터 수를 반환."""
    from sqlalchemy import func

    count = (
        db.session.query(func.count(ClusterResult.id))
        .join(VideoRound, ClusterResult.video_round_id == VideoRound.id)
        .filter(
            VideoRound.session_id == session_id,
            ClusterResult.video_round_id != current_round_id,
            ClusterResult.label == label,
            ClusterResult.category == category,
        )
        .scalar()
    )
    return count or 0


def _first_seen_round(label: str, category: str, session_id: int) -> int | None:
    """동일 label+category가 처음 등장한 라운드 번호 반환."""
    row = (
        db.session.query(VideoRound.round_number)
        .join(ClusterResult, ClusterResult.video_round_id == VideoRound.id)
        .filter(
            VideoRound.session_id == session_id,
            ClusterResult.label == label,
            ClusterResult.category == category,
        )
        .order_by(VideoRound.round_number)
        .first()
    )
    return row[0] if row else None


# ── diff 미리보기 ─────────────────────────────────────────────────────────────


def preview_prompt_change(
    original_prompt: str,
    cluster_results: list[ClusterResult],
    feedbacks_by_cluster: dict[str, list[Feedback]],
) -> dict:
    """
    [팀 직접 구현] 승인된 클러스터를 반영했을 때의 프롬프트 변경 미리보기.

    실제 Claude/LLM 호출 없이 규칙 기반으로 변경 사항을 예측하고
    unified diff를 반환한다.

    Returns:
        {
          "original": str,
          "updated": str,
          "unified_diff": str,
          "change_lines": [{"type": "added"|"removed"|"context", "text": str}],
          "cluster_summaries": [{"label": str, "representative_text": str, "count": int}],
        }
    """
    approved = [cr for cr in cluster_results if cr.status == "approved"]
    if not approved:
        return {
            "original": original_prompt,
            "updated": original_prompt,
            "unified_diff": "",
            "change_lines": [],
            "cluster_summaries": [],
        }

    updated = _build_updated_prompt(original_prompt, approved, feedbacks_by_cluster)
    diff = _make_unified_diff(original_prompt, updated)
    change_lines = _parse_diff_lines(diff)

    return {
        "original": original_prompt,
        "updated": updated,
        "unified_diff": diff,
        "change_lines": change_lines,
        "cluster_summaries": [
            {
                "label": cr.label,
                "representative_text": cr.representative_text,
                "count": cr.count,
                "priority_score": cr.priority_score,
                "scope": cr.scope,
            }
            for cr in approved
        ],
    }


def _build_updated_prompt(
    original: str,
    approved_clusters: list[ClusterResult],
    feedbacks_by_cluster: dict[str, list[Feedback]],
) -> str:
    """[팀 직접 구현] 승인된 클러스터 피드백을 프롬프트에 주석 형태로 추가."""
    lines = [original.rstrip()]
    lines.append("")
    lines.append("# === 시청자 피드백 반영 사항 (자동 생성) ===")

    for cr in approved_clusters:
        scope_tag = "[전역]" if cr.scope == "technical" else "[콘텐츠]"
        lines.append(f"# {scope_tag} {cr.label} (언급 {cr.count}회, 우선순위 {cr.priority_score:.3f})")
        lines.append(f"# 대표 의견: {cr.representative_text}")
        fbs = feedbacks_by_cluster.get(cr.cluster_key, [])
        for fb in fbs[:3]:  # 최대 3개만 표시
            lines.append(f"#   - {fb.comment_text[:80]}")
        lines.append("")

    return "\n".join(lines)


def _make_unified_diff(original: str, updated: str) -> str:
    orig_lines = original.splitlines()
    upd_lines = updated.splitlines()
    diff = difflib.unified_diff(
        orig_lines, upd_lines,
        fromfile="이전 프롬프트",
        tofile="업데이트 프롬프트",
        lineterm="",
    )
    return "\n".join(diff)


def _parse_diff_lines(unified_diff: str) -> list[dict]:
    lines = []
    for line in unified_diff.splitlines():
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            lines.append({"type": "header", "text": line})
        elif line.startswith("+"):
            lines.append({"type": "added", "text": line[1:]})
        elif line.startswith("-"):
            lines.append({"type": "removed", "text": line[1:]})
        else:
            lines.append({"type": "context", "text": line})
    return lines


# ── 승인 적용 ─────────────────────────────────────────────────────────────────


def apply_approved_clusters(
    video_round: VideoRound,
    approved_keys: list[str],
    rejected_keys: list[str],
    deferred_keys: list[str],
    current_prompt: str,
    gemini_api_key: str | None = None,
    gemini_model: str = "gemini-2.0-flash-lite",
    claude_api_key: str | None = None,
    claude_model: str = "claude-sonnet-4-6",
    prompt_engine: str = "auto",
    genre: str = "",
) -> tuple[str, PromptVersion, list[int]]:
    """
    [팀 직접 구현] 승인 결과를 DB에 반영하고 새 PromptVersion을 생성한다.

    Args:
        video_round: 현재 라운드
        approved_keys: 승인된 cluster_key 목록
        rejected_keys: 거절된 cluster_key 목록
        deferred_keys: 보류된 cluster_key 목록
        current_prompt: 현재 프롬프트 텍스트

    Returns:
        (updated_prompt, PromptVersion, shots_to_regenerate)
        shots_to_regenerate: 재생성 필요 샷 번호 목록 (빈 리스트 = 전체 재생성)
    """
    now = datetime.utcnow()

    # 상태 업데이트
    for key, status in [
        *[(k, "approved") for k in approved_keys],
        *[(k, "rejected") for k in rejected_keys],
        *[(k, "deferred") for k in deferred_keys],
    ]:
        cr = ClusterResult.query.filter_by(cluster_key=key).first()
        if cr:
            cr.status = status
            cr.reviewed_at = now

    db.session.flush()

    # 반영 전 인젝션 의심 피드백 자동 플래그
    _flag_injection_suspects(approved_keys)

    # 승인된 클러스터만 모아서 프롬프트 업데이트
    approved_clusters = ClusterResult.query.filter(
        ClusterResult.cluster_key.in_(approved_keys)
    ).all()

    feedbacks_by_cluster: dict[str, list[Feedback]] = {}
    for key in approved_keys:
        feedbacks_by_cluster[key] = Feedback.query.filter_by(cluster_key=key).all()

    # LLM 기반 프롬프트 개선 + 컴포넌트 귀속 (API 키 있을 때)
    component_attribution: list[dict] = []
    _audio_update = ""
    if approved_clusters and (gemini_api_key or claude_api_key) and prompt_engine != "template":
        try:
            from .prompt_updater import update_prompt as llm_update_prompt
            llm_result = llm_update_prompt(
                original_prompt=current_prompt,
                clusters=approved_clusters,
                genre=genre,
                gemini_api_key=gemini_api_key,
                gemini_model=gemini_model,
                api_key=claude_api_key,
                model=claude_model,
                prompt_engine=prompt_engine,
            )
            updated_prompt = llm_result.updated_prompt
            unified_diff = "\n".join(llm_result.diff_lines)
            component_attribution = llm_result.component_attribution
            _audio_update = llm_result.audio_prompt_update or ""
            logger.info("[Approval] LLM 프롬프트 업데이트 완료 (귀속 %d개)", len(component_attribution))
        except Exception as e:
            logger.warning("[Approval] LLM 업데이트 실패, 템플릿 폴백: %s", e)
            updated_prompt = _build_updated_prompt(current_prompt, approved_clusters, feedbacks_by_cluster)
            unified_diff = _make_unified_diff(current_prompt, updated_prompt)
    else:
        updated_prompt = _build_updated_prompt(current_prompt, approved_clusters, feedbacks_by_cluster)
        unified_diff = _make_unified_diff(current_prompt, updated_prompt)

    # PromptVersion 생성
    version_number = _next_version_number(video_round.session_id)
    pv = PromptVersion(
        session_id=video_round.session_id,
        video_round_id=video_round.id,
        version_number=version_number,
        prompt_text=updated_prompt,
        unified_diff=unified_diff,
        change_summary=_build_change_summary(approved_clusters),
        component_attribution_json=json.dumps(component_attribution, ensure_ascii=False) if component_attribution else None,
    )
    pv.set_applied_cluster_keys(approved_keys)
    db.session.add(pv)

    # 승인된 클러스터의 피드백 → is_reflected = True
    for key in approved_keys:
        Feedback.query.filter_by(cluster_key=key).update({"is_reflected": True})

    # technical 클러스터 → GlobalTechnicalSetting 승격
    for cr in approved_clusters:
        if cr.scope == "technical":
            _promote_to_global(cr, video_round)

    # audio 클러스터 중 voice 관련 → VoiceSettings 승격
    _promote_voice_settings(approved_clusters, video_round.session_id)

    # audio 클러스터 중 BGM 볼륨/스타일 → audio 설정 승격
    _promote_bgm_settings(approved_clusters, video_round.session_id, _audio_update)

    # audio 클러스터 중 나레이션 볼륨 피드백 → narration_db 승격
    _promote_narration_volume(approved_clusters, video_round.session_id)

    # 나레이션 내용/분량 피드백 → 자동 반영 불가, needs_review 마킹
    _flag_narration_content_review(approved_clusters)

    # 로고/자막 피드백 → video_post 플래그 승격
    _promote_visual_settings(approved_clusters, video_round.session_id)

    # 로고 아웃트로 활성화 시 → 다음 라운드 Kling 프롬프트에 클로징 샷 힌트 추가
    _logo_flag = GlobalTechnicalSetting.query.filter_by(
        source_session_id=video_round.session_id,
        category="video_post",
        setting_key="logo_outro",
        is_active=True,
    ).first()
    _CLOSING_HINT = (
        "\nClosing: Wide establishing shot, camera slowly pulls back, "
        "calm and dignified ending frame."
    )
    if _logo_flag and _CLOSING_HINT.strip() not in updated_prompt:
        updated_prompt += _CLOSING_HINT
        pv.prompt_text = updated_prompt

    db.session.flush()

    # 분류기 온라인 학습: 승인된 피드백으로 constructive 프로토타입 업데이트
    _update_classifier_prototypes(approved_keys)

    # [샷 단위 피드백] 재생성이 필요한 샷 번호 목록 계산
    from ..pipeline.shot_mapper import get_shots_needing_regeneration
    from ..models import ShotRound
    shots = ShotRound.query.filter_by(video_round_id=video_round.id).all()
    shots_to_regenerate = get_shots_needing_regeneration(approved_keys, shots)

    logger.info(
        "승인 적용: 승인 %d, 거절 %d, 보류 %d → 버전 v%d / 재생성 샷: %s",
        len(approved_keys), len(rejected_keys), len(deferred_keys), version_number,
        shots_to_regenerate or "전체",
    )
    return updated_prompt, pv, shots_to_regenerate


def _next_version_number(session_id: int) -> int:
    """세션 내 최대 버전 번호 + 1 반환."""
    from sqlalchemy import func

    max_v = (
        db.session.query(func.max(PromptVersion.version_number))
        .filter_by(session_id=session_id)
        .scalar()
    )
    return (max_v or 0) + 1


def _build_change_summary(approved_clusters: list[ClusterResult]) -> str:
    if not approved_clusters:
        return "변경 없음"
    items = [f"{cr.label}(×{cr.count})" for cr in approved_clusters[:5]]
    tail = f" 외 {len(approved_clusters)-5}건" if len(approved_clusters) > 5 else ""
    return "반영: " + ", ".join(items) + tail


def _promote_to_global(cr: ClusterResult, video_round: VideoRound) -> None:
    """[팀 직접 구현] technical 클러스터를 GlobalTechnicalSetting으로 승격."""
    setting_key = f"{cr.category}_{cr.cluster_key[:8]}"
    existing = GlobalTechnicalSetting.query.filter_by(
        category=cr.category, setting_key=setting_key
    ).first()
    if existing:
        existing.setting_value = cr.representative_text
        existing.updated_at = datetime.utcnow()
        existing.is_active = True
    else:
        setting = GlobalTechnicalSetting(
            category=cr.category,
            setting_key=setting_key,
            setting_value=cr.representative_text,
            description=cr.label,
            source_cluster_key=cr.cluster_key,
            source_session_id=video_round.session_id,
            source_round_id=video_round.id,
        )
        db.session.add(setting)


# ── 버전 롤백 ─────────────────────────────────────────────────────────────────


def rollback_to_version(
    target_version: PromptVersion,
    current_round: VideoRound,
) -> PromptVersion:
    """
    [팀 직접 구현] 지정된 버전의 프롬프트로 롤백.

    롤백 자체를 새 버전으로 기록하므로 forward/undo가 모두 가능.

    Args:
        target_version: 되돌릴 PromptVersion
        current_round: 현재 VideoRound

    Returns:
        새로 생성된 PromptVersion (롤백 기록)
    """
    current_pv = (
        PromptVersion.query
        .filter_by(session_id=target_version.session_id)
        .order_by(PromptVersion.version_number.desc())
        .first()
    )

    current_text = current_pv.prompt_text if current_pv else current_round.video_gen_prompt or ""
    current_version_num = current_pv.version_number if current_pv else 0

    diff = _make_unified_diff(current_text, target_version.prompt_text)
    new_version_number = _next_version_number(target_version.session_id)

    rollback_pv = PromptVersion(
        session_id=target_version.session_id,
        video_round_id=current_round.id,
        version_number=new_version_number,
        prompt_text=target_version.prompt_text,
        unified_diff=diff,
        change_summary=f"v{current_version_num} → v{target_version.version_number}으로 롤백",
        rolled_back_from_version=current_version_num,
    )
    db.session.add(rollback_pv)

    # 현재 라운드 프롬프트 업데이트
    current_round.video_gen_prompt = target_version.prompt_text
    current_round.prompt_updated_at = datetime.utcnow()

    db.session.flush()
    logger.info("롤백: v%d → v%d (새 기록 v%d)", current_version_num, target_version.version_number, new_version_number)
    return rollback_pv


# ── 인젝션 의심 피드백 플래그 ────────────────────────────────────────────────


def _flag_injection_suspects(approved_keys: list[str]) -> None:
    """
    [팀 직접 구현] 승인 대상 피드백 중 인젝션 의심 패턴이 있는 것을 needs_review=True로 표시.

    유튜브 댓글은 신뢰할 수 없는 외부 입력이므로, LLM에 전달되기 전에
    지시문 주입 시도를 감지해 수동 검토 대상으로 분류한다.
    """
    flagged = 0
    for key in approved_keys:
        feedbacks = Feedback.query.filter_by(cluster_key=key).all()
        for fb in feedbacks:
            if fb.comment_text and _detect_injection(fb.comment_text):
                fb.needs_review = True
                flagged += 1
    if flagged:
        logger.warning("[Approval] 인젝션 의심 피드백 %d개 needs_review=True 설정", flagged)


# ── 분류기 온라인 학습 ───────────────────────────────────────────────────────


def _update_classifier_prototypes(approved_keys: list[str]) -> None:
    """
    [팀 직접 구현] 승인된 클러스터 피드백으로 EmbeddingClassifier 프로토타입을 업데이트.

    승인 = "이 텍스트들은 진짜 건설적 피드백이다"는 인간 레이블.
    weight=0.1: 한 라운드가 기존 프로토타입을 최대 10%만 바꿈 → 점진적·안정적 적응.
    실패해도 예외를 던지지 않아 메인 승인 흐름에 영향 없음.
    """
    if not approved_keys:
        return
    try:
        from .classifier.factory import get_classifier
        from .classifier.embedding_classifier import EmbeddingClassifier

        classifier = get_classifier()
        if not isinstance(classifier, EmbeddingClassifier):
            return

        texts = [
            fb.comment_text
            for key in approved_keys
            for fb in Feedback.query.filter_by(cluster_key=key).all()
            if fb.comment_text
        ]
        if texts:
            classifier.update_prototype("constructive", texts, weight=0.1)
            logger.info("분류기 온라인 학습 완료: constructive +%d개 피드백", len(texts))
    except Exception as e:
        logger.warning("분류기 프로토타입 업데이트 건너뜀 (무시): %s", e)


# ── 재발 확인 ─────────────────────────────────────────────────────────────────


def check_reoccurrence(
    session_id: int,
    current_round_number: int,
) -> list[dict]:
    """
    [팀 직접 구현] 현재 라운드에서 이전 라운드에도 등장했던 클러스터 목록 반환.

    반복 등장하는 피드백 = 이전에 반영됐지만 효과가 없었거나
    반영이 안 된 중요 의견일 가능성이 높다.

    Returns:
        [{"label": str, "category": str, "reoccurrence_count": int, "first_seen_round": int}]
    """
    results = (
        db.session.query(ClusterResult)
        .join(VideoRound, ClusterResult.video_round_id == VideoRound.id)
        .filter(
            VideoRound.session_id == session_id,
            VideoRound.round_number == current_round_number,
            ClusterResult.reoccurrence_count > 0,
        )
        .order_by(ClusterResult.reoccurrence_count.desc())
        .all()
    )

    return [
        {
            "label": cr.label,
            "category": cr.category,
            "reoccurrence_count": cr.reoccurrence_count,
            "first_seen_round": cr.first_seen_round,
            "priority_score": cr.priority_score,
        }
        for cr in results
    ]


# ── 음성 피드백 → VoiceSettings 승격 ──────────────────────────────────────────

_VOICE_KEYWORDS    = ["목소리", "음성", "나레이션", "내레이션", "보이스", "voice", "narration", "tts", "ai음성", "ai 음성"]
_BGM_KEYWORDS      = ["배경음악", "bgm", "효과음", "음악", "music", "볼륨", "volume", "soundtrack"]
_LOGO_KEYWORDS     = ["로고", "logo", "마크", "엠블럼", "브랜딩", "학교 로고", "학과 로고", "학교 표시", "학교 마크"]
_SUBTITLE_KEYWORDS = ["자막", "subtitle", "자막도", "글자", "텍스트 표시", "텍스트도", "cc"]
_FADE_KEYWORDS     = ["자연스럽게 끌", "자연스럽게 끝", "부드럽게 끝", "끝나는 부분",
                      "마지막 장면", "엔딩을", "fade", "페이드", "천천히 끝", "여운"]
_FADE_DURATION_EXTENDED = 2.5  # 피드백으로 확장된 페이드 길이(초)


def _classify_audio_cluster(text: str) -> str:
    lower = text.lower()
    voice_score = sum(1 for kw in _VOICE_KEYWORDS if kw in lower)
    bgm_score   = sum(1 for kw in _BGM_KEYWORDS   if kw in lower)
    if voice_score == 0 and bgm_score == 0:
        return "voice"
    if voice_score == bgm_score:
        # 동점 시: BGM 볼륨 조절 액션 키워드가 있으면 BGM 피드백으로 분류
        # (예: "음악을 줄이거나 나레이션을 키워주세요" → bgm)
        has_bgm_action = any(kw in lower for kw in _BGM_LOUDER_KEYWORDS + _BGM_QUIETER_KEYWORDS)
        return "bgm" if has_bgm_action else "voice"
    return "voice" if voice_score > bgm_score else "bgm"


def _promote_voice_settings(approved_clusters: list, session_id: int) -> None:
    """
    [팀 직접 구현] 승인된 audio 클러스터 중 voice 관련 것들을 분석해
    GlobalTechnicalSetting에 최적 voice_id를 저장한다.

    저장 키:
      - category="voice", setting_key="voice_id"   → 선택된 보이스 ID
      - category="voice", setting_key="voice_reason" → 변경 이유 (UI 표시용)
    """
    audio_clusters = [c for c in approved_clusters if c.category == "audio"]
    voice_clusters = [
        c for c in audio_clusters
        if _classify_audio_cluster(c.representative_text or c.label or "") == "voice"
    ]
    if not voice_clusters:
        return

    # 피드백 텍스트 수집
    texts: list[str] = []
    for cr in voice_clusters:
        if cr.representative_text:
            texts.append(cr.representative_text)
        fbs = Feedback.query.filter_by(cluster_key=cr.cluster_key).all()
        texts.extend(fb.comment_text for fb in fbs if fb.comment_text)

    # 현재 voice_id 조회 (이전 라운드 선택값 또는 기본값)
    prev_setting = GlobalTechnicalSetting.query.filter_by(
        source_session_id=session_id,
        category="voice",
        setting_key="voice_id",
    ).first()
    from ..models import PipelineSession
    sess = PipelineSession.query.get(session_id)
    from flask import current_app
    default_voice_id = current_app.config.get(
        "ELEVENLABS_NARRATION_VOICE_ID", "pNInz6obpgDQGcFmaJgB"
    )
    current_voice_id = prev_setting.setting_value if prev_setting else default_voice_id

    # 보이스 선택 + 감정 프로파일 도출
    from .voice_selector import select_voice, get_emotion_profile
    selected_id, reason, analysis = select_voice(texts, current_voice_id)
    emotion_profile = get_emotion_profile(texts)

    def _upsert(key: str, value: str, description: str) -> None:
        row = GlobalTechnicalSetting.query.filter_by(
            source_session_id=session_id,
            category="voice",
            setting_key=key,
        ).first()
        if row:
            row.setting_value = value
            row.is_active = True
        else:
            db.session.add(GlobalTechnicalSetting(
                category="voice",
                setting_key=key,
                setting_value=value,
                description=description,
                source_session_id=session_id,
            ))

    _upsert("voice_id", selected_id, f"피드백 기반 자동 선택 — {reason}")
    _upsert("voice_reason", reason, "보이스 변경 이유")
    _upsert(
        "emotion_profile",
        json.dumps(emotion_profile, ensure_ascii=False),
        "큐 위치별 감정 태그 프로파일",
    )

    logger.info(
        "[Approval] voice 선택 완료: session=%d, voice_id=%s, reason=%s",
        session_id, selected_id, reason,
    )


# ── BGM 피드백 → audio 설정 승격 ─────────────────────────────────────────────

_BGM_LOUDER_KEYWORDS  = ["너무 커", "시끄럽", "bgm이 커", "음악이 커", "배경음이 커",
                          "목소리가 안 들", "목소리가 묻", "나레이션이 안 들",
                          "줄여", "낮춰", "줄이", "낮추", "난잡", "겹쳐"]
_BGM_QUIETER_KEYWORDS = ["너무 작", "잘 안 들", "bgm이 작", "음악이 작", "배경음이 작",
                          "더 크게", "높여", "올려"]
_BGM_STEP_DB = 6.0   # 볼륨 조정 단계 (dB)
_BGM_DB_MIN  = -36.0
_BGM_DB_MAX  = -6.0
_BGM_DB_DEFAULT = -18.0


def _promote_bgm_settings(approved_clusters: list, session_id: int, audio_prompt_update: str) -> None:
    """
    [팀 직접 구현] BGM 관련 피드백을 GlobalTechnicalSetting에 저장.

    저장 키:
      - category="audio", setting_key="bgm_db"             → BGM 볼륨 (dB 문자열)
      - category="audio", setting_key="bgm_prompt_override" → BGM 스타일 텍스트
    """
    audio_clusters = [c for c in approved_clusters if c.category == "audio"]
    bgm_clusters = [
        c for c in audio_clusters
        if _classify_audio_cluster(c.representative_text or c.label or "") == "bgm"
    ]

    parts: list[str] = []
    for c in bgm_clusters:
        if c.representative_text:
            parts.append(c.representative_text)
        if c.label:
            parts.append(c.label)
        fbs = Feedback.query.filter_by(cluster_key=c.cluster_key).all()
        parts.extend(fb.comment_text for fb in fbs if fb.comment_text)

    combined = " ".join(parts).lower()

    louder_hit  = any(kw in combined for kw in _BGM_LOUDER_KEYWORDS)
    quieter_hit = any(kw in combined for kw in _BGM_QUIETER_KEYWORDS)

    def _save(key: str, value: str, description: str) -> None:
        row = GlobalTechnicalSetting.query.filter_by(
            source_session_id=session_id, category="audio", setting_key=key,
        ).first()
        if row:
            row.setting_value = value
            row.is_active = True
        else:
            db.session.add(GlobalTechnicalSetting(
                category="audio", setting_key=key,
                setting_value=value, description=description,
                source_session_id=session_id,
            ))

    # BGM 볼륨 조정
    if louder_hit or quieter_hit:
        existing = GlobalTechnicalSetting.query.filter_by(
            source_session_id=session_id, category="audio", setting_key="bgm_db",
        ).first()
        current_db = float(existing.setting_value) if existing else _BGM_DB_DEFAULT
        if louder_hit:
            new_db = max(_BGM_DB_MIN, current_db - _BGM_STEP_DB)
            reason = "BGM 볼륨 낮춤 (피드백: 나레이션과 겹침)"
        else:
            new_db = min(_BGM_DB_MAX, current_db + _BGM_STEP_DB)
            reason = "BGM 볼륨 높임 (피드백: 너무 작음)"
        _save("bgm_db", str(new_db), reason)
        logger.info("[Approval] bgm_db 조정: %.0f → %.0fdB (session=%d)", current_db, new_db, session_id)

    # BGM 스타일 텍스트 저장 (ElevenLabs 프롬프트 오버라이드)
    if audio_prompt_update:
        _save("bgm_prompt_override", audio_prompt_update[:400], "피드백 기반 BGM 스타일 오버라이드")
        logger.info("[Approval] bgm_prompt_override 저장: session=%d", session_id)


# ── 나레이션 볼륨 피드백 → audio 설정 승격 ───────────────────────────────────

_NARRATION_LOUDER_KEYWORDS = [
    "나레이션이 너무 작", "목소리가 너무 작", "나레이션 볼륨 높", "목소리 볼륨 높",
    "나레이션 소리 높", "나레이션이 잘 안 들", "목소리가 잘 안 들",
    "나레이션을 키", "목소리를 키", "나레이션 크게", "목소리 크게",
    "나레이션소리를 키", "소리를 키",
]
_NARRATION_QUIETER_KEYWORDS = [
    "나레이션이 너무 커", "목소리가 너무 커", "나레이션 볼륨 낮", "목소리 볼륨 낮",
    "나레이션 소리 낮", "나레이션을 낮", "목소리를 낮", "나레이션 작게", "목소리 작게",
]
_NARRATION_STEP_DB   = 3.0
_NARRATION_DB_MIN    = -6.0
_NARRATION_DB_MAX    = 12.0
_NARRATION_DB_DEFAULT = 0.0


def _promote_narration_volume(approved_clusters: list, session_id: int) -> None:
    """
    [팀 직접 구현] 나레이션 볼륨 피드백을 GlobalTechnicalSetting에 저장.

    저장 키:
      - category="audio", setting_key="narration_db" → 나레이션 볼륨 조정값 (dB 문자열)
    """
    audio_clusters = [c for c in approved_clusters if c.category == "audio"]
    voice_clusters = [
        c for c in audio_clusters
        if _classify_audio_cluster(c.representative_text or c.label or "") == "voice"
    ]

    parts: list[str] = []
    for c in voice_clusters:
        if c.representative_text:
            parts.append(c.representative_text)
        if c.label:
            parts.append(c.label)
        fbs = Feedback.query.filter_by(cluster_key=c.cluster_key).all()
        parts.extend(fb.comment_text for fb in fbs if fb.comment_text)

    combined = " ".join(parts).lower()

    louder_hit  = any(kw in combined for kw in _NARRATION_LOUDER_KEYWORDS)
    quieter_hit = any(kw in combined for kw in _NARRATION_QUIETER_KEYWORDS)

    if not louder_hit and not quieter_hit:
        return

    existing = GlobalTechnicalSetting.query.filter_by(
        source_session_id=session_id, category="audio", setting_key="narration_db",
    ).first()
    current_db = float(existing.setting_value) if existing else _NARRATION_DB_DEFAULT
    if louder_hit:
        new_db = min(_NARRATION_DB_MAX, current_db + _NARRATION_STEP_DB)
        reason = "나레이션 볼륨 높임 (피드백: 목소리가 잘 안 들림)"
    else:
        new_db = max(_NARRATION_DB_MIN, current_db - _NARRATION_STEP_DB)
        reason = "나레이션 볼륨 낮춤 (피드백: 목소리가 너무 큼)"

    row = GlobalTechnicalSetting.query.filter_by(
        source_session_id=session_id, category="audio", setting_key="narration_db",
    ).first()
    if row:
        row.setting_value = str(new_db)
        row.is_active = True
    else:
        db.session.add(GlobalTechnicalSetting(
            category="audio", setting_key="narration_db",
            setting_value=str(new_db), description=reason,
            source_session_id=session_id,
        ))
    logger.info("[Approval] narration_db 조정: %.0f → %.0fdB (session=%d)", current_db, new_db, session_id)


# ── 나레이션 내용 피드백 → needs_review 마킹 ──────────────────────────────────

_NARRATION_CONTENT_KEYWORDS = [
    "나레이션 양", "말하는 양", "멘트 추가", "멘트 늘", "말이 적", "말이 많",
    "대사 추가", "대사 늘", "더 말해", "나레이션 추가", "내용 추가", "설명 추가",
    "더 설명", "말 좀 더", "얘기가 적", "나레이션이 적", "설명이 적",
    "나레이션 줄", "말이 너무 많", "말이 너무 길",
]


def _flag_narration_content_review(approved_clusters: list) -> None:
    """
    [팀 직접 구현] 나레이션 내용/분량 피드백은 자동 반영 불가.
    해당 피드백을 needs_review=True로 마킹해 클러스터 검토 페이지에서
    '직접 수정 필요' 항목으로 표시되도록 한다.
    """
    for c in approved_clusters:
        if c.category != "audio":
            continue
        text = ((c.representative_text or "") + " " + (c.label or "")).lower()
        if not any(kw in text for kw in _NARRATION_CONTENT_KEYWORDS):
            continue
        count = Feedback.query.filter_by(cluster_key=c.cluster_key).update({"needs_review": True})
        logger.info(
            "[Approval] 나레이션 내용 피드백 needs_review 마킹 (%d개): %s", count, c.label
        )


# ── 시각 피드백 → video_post 플래그 승격 ──────────────────────────────────────


def _promote_visual_settings(approved_clusters: list, session_id: int) -> None:
    """
    [팀 직접 구현] 승인된 클러스터에서 로고/자막 관련 피드백을 감지해
    GlobalTechnicalSetting에 플래그를 저장한다.

    저장 키:
      - category="video_post", setting_key="logo_outro"    → "true"
      - category="video_post", setting_key="subtitle_auto" → "true"
    """
    # 클러스터 대표 텍스트 + 라벨 + 실제 피드백 댓글 전체 수집
    parts: list[str] = []
    for c in approved_clusters:
        if c.representative_text:
            parts.append(c.representative_text)
        if c.label:
            parts.append(c.label)
        fbs = Feedback.query.filter_by(cluster_key=c.cluster_key).all()
        parts.extend(fb.comment_text for fb in fbs if fb.comment_text)

    combined = " ".join(parts).lower()

    logo_hit     = any(kw in combined for kw in _LOGO_KEYWORDS)
    subtitle_hit = any(kw in combined for kw in _SUBTITLE_KEYWORDS)

    if not logo_hit and not subtitle_hit:
        return

    def _flag(key: str, description: str, value: str = "true") -> None:
        row = GlobalTechnicalSetting.query.filter_by(
            source_session_id=session_id,
            category="video_post",
            setting_key=key,
        ).first()
        if row:
            row.setting_value = value
            row.is_active = True
        else:
            db.session.add(GlobalTechnicalSetting(
                category="video_post",
                setting_key=key,
                setting_value=value,
                description=description,
                source_session_id=session_id,
            ))

    fade_hit = any(kw in combined for kw in _FADE_KEYWORDS)

    if logo_hit:
        _flag("logo_outro", "피드백 기반 로고 카드 아웃트로 활성화")
        logger.info("[Approval] logo_outro 플래그 설정: session=%d", session_id)

    if subtitle_hit:
        _flag("subtitle_auto", "피드백 기반 자막 자동 생성 활성화")
        logger.info("[Approval] subtitle_auto 플래그 설정: session=%d", session_id)

    if fade_hit:
        _flag("fade_duration", "피드백 기반 페이드 길이 확장", str(_FADE_DURATION_EXTENDED))
        logger.info(
            "[Approval] fade_duration 확장 (%.1fs): session=%d",
            _FADE_DURATION_EXTENDED, session_id,
        )

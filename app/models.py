import json
import secrets
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PipelineSession(db.Model):
    __tablename__ = "pipeline_session"

    id = db.Column(db.Integer, primary_key=True)
    genre = db.Column(db.String(100), nullable=False)
    title = db.Column(db.String(200))
    user_script = db.Column(db.Text, nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    # H-4: 소유자 인증 토큰 — 데모 환경에서는 NULL (쿠키 유실로 인한 권한 오류 방지)
    owner_token = db.Column(db.String(64), nullable=True)
    canvas_json = db.Column(db.Text, nullable=True)   # Next.js 캔버스 직렬화 JSON

    rounds = db.relationship(
        "VideoRound",
        backref="session",
        lazy=True,
        order_by="VideoRound.round_number",
    )
    metrics = db.relationship("Metric", backref="session", lazy=True)

    @property
    def current_round(self):
        """가장 최근 라운드 (round_number 기준 마지막). 없으면 None."""
        return self.rounds[-1] if self.rounds else None

    def to_dict(self):
        return {
            "id": self.id,
            "genre": self.genre,
            "title": self.title,
            "user_script": self.user_script,
            "description": self.description,
            "created_at": self.created_at.isoformat(),
            "round_count": len(self.rounds),
        }


class VideoRound(db.Model):
    __tablename__ = "video_round"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer, db.ForeignKey("pipeline_session.id"), nullable=False
    )
    round_number = db.Column(db.Integer, nullable=False)

    # 파이프라인 진행 상태
    # prompt_built → video_ready → uploaded → feedback_collected →
    # feedback_classified → feedback_clustered → pending_review → prompt_updated
    status = db.Column(db.String(50), default="prompt_built")

    video_gen_prompt = db.Column(db.Text)
    storyboard = db.Column(db.Text)
    video_file_path = db.Column(db.String(500))
    youtube_video_id = db.Column(db.String(100))
    youtube_url = db.Column(db.String(500))

    # 시간 측정 (반영 소요 시간 단축 효과 계산용)
    pipeline_start_at = db.Column(db.DateTime)
    prompt_updated_at = db.Column(db.DateTime)

    # 영상 생성 어댑터 및 예상 비용 추적 (일일 한도·과금 관리용)
    generator_adapter = db.Column(db.String(20))       # veo | runway | mock
    estimated_cost_usd = db.Column(db.Float, default=0.0)

    # [팀 직접 구현] 음악 생성 추적 — ElevenLabs Music / Sound 어댑터 결과 저장
    music_file_path = db.Column(db.String(500))
    music_adapter = db.Column(db.String(30))           # elevenlabs_music | elevenlabs_sound | mock_music
    music_prompt = db.Column(db.Text)                  # 음악 생성 시 사용된 프롬프트 (피드백 반영 추적용)
    estimated_music_cost_usd = db.Column(db.Float, default=0.0)

    # [팀 직접 구현] 립싱크 추적 — SadTalker / Wav2Lip 어댑터 결과 저장
    lipsync_file_path = db.Column(db.String(500))
    lipsync_adapter = db.Column(db.String(20))         # sadtalker | wav2lip | mock
    character_image_path = db.Column(db.String(500))   # 립싱크 입력 캐릭터 이미지

    # [팀 직접 구현] 자막·메타데이터 추적
    srt_file_path = db.Column(db.String(500))           # Whisper 생성 SRT 경로
    youtube_metadata_json = db.Column(db.Text)          # Gemini 생성 YouTube 메타데이터 (title/desc/tags)

    # [팀 직접 구현] 프레임 체이닝 — 다음 라운드 Veo image_url 참조용 마지막 프레임 경로
    last_frame_path = db.Column(db.String(500))

    # F-1: 프롬프트 성능 메모리 — 이 라운드의 최종 피드백 결과 기록
    positive_feedback_ratio = db.Column(db.Float, default=0.0)  # 긍정 피드백 비율
    total_feedback_count = db.Column(db.Integer, default=0)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    feedbacks = db.relationship("Feedback", backref="video_round", lazy=True)
    cluster_results = db.relationship("ClusterResult", backref="video_round", lazy=True)
    prompt_change_from = db.relationship(
        "PromptChange",
        backref="from_round",
        foreign_keys="PromptChange.from_round_id",
        uselist=False,
    )

    def constructive_feedbacks(self):
        return [f for f in self.feedbacks if f.category == "constructive"]

    def reflected_feedbacks(self):
        return [f for f in self.feedbacks if f.is_reflected]

    def reflection_rate(self):
        constructive = self.constructive_feedbacks()
        if not constructive:
            return 0.0
        reflected = [f for f in constructive if f.is_reflected]
        return len(reflected) / len(constructive)

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "round_number": self.round_number,
            "status": self.status,
            "video_gen_prompt": self.video_gen_prompt,
            "storyboard": self.storyboard,
            "youtube_video_id": self.youtube_video_id,
            "youtube_url": self.youtube_url,
            "generator_adapter": self.generator_adapter,
            "estimated_cost_usd": self.estimated_cost_usd,
            "music_file_path": self.music_file_path,
            "music_adapter": self.music_adapter,
            "music_prompt": self.music_prompt,
            "estimated_music_cost_usd": self.estimated_music_cost_usd,
            "lipsync_file_path": self.lipsync_file_path,
            "lipsync_adapter": self.lipsync_adapter,
            "character_image_path": self.character_image_path,
            "created_at": self.created_at.isoformat(),
            "total_feedbacks": len(self.feedbacks),
            "constructive_count": len(self.constructive_feedbacks()),
            "reflected_count": len(self.reflected_feedbacks()),
            "reflection_rate": self.reflection_rate(),
        }


class Feedback(db.Model):
    __tablename__ = "feedback"

    id = db.Column(db.Integer, primary_key=True)
    video_round_id = db.Column(
        db.Integer, db.ForeignKey("video_round.id"), nullable=False
    )

    # 출처: 'youtube' | 'simulated' | 'test'
    source = db.Column(db.String(50), default="youtube")
    youtube_comment_id = db.Column(db.String(100))
    author_name = db.Column(db.String(200))
    comment_text = db.Column(db.Text, nullable=False)

    # 분류 결과 (classifier 출력)
    # constructive | non_constructive | spam | inappropriate
    category = db.Column(db.String(50))
    improvement_suggestion = db.Column(db.Text)
    confidence = db.Column(db.Float)
    classification_reasoning = db.Column(db.Text)

    # 2축 판단 점수 (classifier 출력)
    specificity_score = db.Column(db.Float)   # 구체성 (0~1)
    validity_score = db.Column(db.Float)      # 타당성 (0~1)

    # [v2 신규] 확장 3축 점수 (classifier 출력)
    severity_score = db.Column(db.Float)      # 심각도 (0~1)
    urgency_score = db.Column(db.Float)       # 긴급도 (0~1)
    quality_score = db.Column(db.Float)       # 피드백 품질 종합 (0~1)

    # 주제 분류 (clusterer 입력)
    # story | visual | audio | subtitle | pacing | content | technical | other
    topic_category = db.Column(db.String(50))
    # [v2 신규] 멀티레이블 토픽 (JSON: [(category, score), ...])
    topic_categories_json = db.Column(db.Text, default="[]")
    # content | technical
    scope = db.Column(db.String(20))
    # 감지된 언어 (auto | ko | en | ja | mixed_ko_en | ...)
    detected_language = db.Column(db.String(20), default="auto")

    # [v2 신규] 수동 검토 권장 플래그 (신뢰도가 낮은 경우 True)
    needs_review = db.Column(db.Boolean, default=False)
    # F-2: 실행가능성 축 — 구체적 개선 제안 포함 여부 (0=추상적, 1=구체적)
    actionability_score = db.Column(db.Float, default=0.5)

    # 클러스터링 (cluster_key → ClusterResult.cluster_key)
    cluster_key = db.Column(db.String(36))    # UUID (ClusterResult.cluster_key)
    cluster_id = db.Column(db.Integer)        # 하위 호환용 (레거시)
    cluster_label = db.Column(db.String(200))
    embedding_json = db.Column(db.Text)

    # [샷 단위 피드백] 댓글의 타임스탬프 → 특정 샷에 매핑
    timestamp_seconds = db.Column(db.Float, nullable=True)       # 댓글에서 추출한 타임스탬프(초)
    shot_round_id = db.Column(db.Integer, db.ForeignKey("shot_round.id"), nullable=True)

    # 반영 추적
    is_reflected = db.Column(db.Boolean, default=False)
    reflection_notes = db.Column(db.Text)

    collected_at = db.Column(db.DateTime, default=datetime.utcnow)
    classified_at = db.Column(db.DateTime)

    def get_embedding(self):
        if self.embedding_json:
            return json.loads(self.embedding_json)
        return None

    def set_embedding(self, vec):
        self.embedding_json = json.dumps(vec)

    def get_topic_categories(self) -> list:
        return json.loads(self.topic_categories_json or "[]")

    def to_dict(self):
        return {
            "id": self.id,
            "video_round_id": self.video_round_id,
            "source": self.source,
            "author_name": self.author_name,
            "comment_text": self.comment_text,
            "category": self.category,
            "improvement_suggestion": self.improvement_suggestion,
            "confidence": self.confidence,
            "specificity_score": self.specificity_score,
            "validity_score": self.validity_score,
            "severity_score": self.severity_score,
            "urgency_score": self.urgency_score,
            "quality_score": self.quality_score,
            "topic_category": self.topic_category,
            "topic_categories": self.get_topic_categories(),
            "scope": self.scope,
            "detected_language": self.detected_language,
            "needs_review": self.needs_review,
            "actionability_score": self.actionability_score,
            "cluster_key": self.cluster_key,
            "cluster_label": self.cluster_label,
            "is_reflected": self.is_reflected,
            "reflection_notes": self.reflection_notes,
            "collected_at": self.collected_at.isoformat() if self.collected_at else None,
        }


class ClusterResult(db.Model):
    """
    [팀 직접 설계] 클러스터링 결과 저장 + 사용자 승인 워크플로.

    status 흐름:
      pending → approved (프롬프트 반영됨)
                rejected (사용자 거절)
                deferred (나중에 검토)
    """
    __tablename__ = "cluster_result"

    id = db.Column(db.Integer, primary_key=True)
    video_round_id = db.Column(
        db.Integer, db.ForeignKey("video_round.id"), nullable=False
    )

    # UUID (clusterer가 생성한 키, Feedback.cluster_key와 매핑)
    cluster_key = db.Column(db.String(36), unique=True, nullable=False)

    # 주제 및 범위
    category = db.Column(db.String(50), nullable=False)
    scope = db.Column(db.String(20), nullable=False)  # content | technical

    # 자동 레이블 및 대표 텍스트
    label = db.Column(db.String(200))
    representative_text = db.Column(db.Text)

    # 클러스터 통계
    count = db.Column(db.Integer, default=0)           # 피드백 수
    avg_similarity = db.Column(db.Float)               # 평균 코사인 유사도
    priority_score = db.Column(db.Float)               # 우선순위 점수 (5축 공식)
    freq_score = db.Column(db.Float)                   # count / total_constructive
    avg_validity_score = db.Column(db.Float)           # 평균 타당성 점수
    contradicted_by_key = db.Column(db.String(36))     # 상반된 클러스터 key

    # [v2 신규] 확장 통계
    avg_severity_score = db.Column(db.Float)           # 평균 심각도 (0~1)
    avg_urgency_score = db.Column(db.Float)            # 평균 긴급도 (0~1)
    avg_quality_score = db.Column(db.Float)            # 평균 피드백 품질 (0~1)
    cohesion_score = db.Column(db.Float)               # 클러스터 응집도 (0~1)
    cluster_summary = db.Column(db.Text)               # 추출적 요약 문장

    # 사용자 승인 워크플로
    # pending | approved | rejected | deferred
    status = db.Column(db.String(20), default="pending")
    reviewed_at = db.Column(db.DateTime)
    review_notes = db.Column(db.Text)

    # [샷 단위 피드백] 이 클러스터가 타깃하는 샷 번호 (None = 전체 영상)
    target_shot_number = db.Column(db.Integer, nullable=True)

    # 재발 추적 (이전 라운드에서도 같은 클러스터가 등장했는지)
    reoccurrence_count = db.Column(db.Integer, default=0)
    first_seen_round = db.Column(db.Integer)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_feedback_ids(self, session):
        """이 클러스터에 속한 Feedback id 목록 반환."""
        feedbacks = session.query(Feedback).filter_by(cluster_key=self.cluster_key).all()
        return [f.id for f in feedbacks]

    def to_dict(self):
        return {
            "id": self.id,
            "video_round_id": self.video_round_id,
            "cluster_key": self.cluster_key,
            "category": self.category,
            "scope": self.scope,
            "label": self.label,
            "representative_text": self.representative_text,
            "cluster_summary": self.cluster_summary,
            "count": self.count,
            "avg_similarity": self.avg_similarity,
            "cohesion_score": self.cohesion_score,
            "priority_score": self.priority_score,
            "freq_score": self.freq_score,
            "avg_validity_score": self.avg_validity_score,
            "avg_severity_score": self.avg_severity_score,
            "avg_urgency_score": self.avg_urgency_score,
            "avg_quality_score": self.avg_quality_score,
            "contradicted_by_key": self.contradicted_by_key,
            "status": self.status,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "review_notes": self.review_notes,
            "target_shot_number": self.target_shot_number,
            "reoccurrence_count": self.reoccurrence_count,
            "first_seen_round": self.first_seen_round,
            "created_at": self.created_at.isoformat(),
        }


class PromptVersion(db.Model):
    """
    [팀 직접 설계] 프롬프트 버전 관리.

    매 프롬프트 변경 시 새 레코드 생성.
    unified diff로 이전 버전과 비교 가능.
    """
    __tablename__ = "prompt_version"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer, db.ForeignKey("pipeline_session.id"), nullable=False
    )
    video_round_id = db.Column(
        db.Integer, db.ForeignKey("video_round.id"), nullable=False
    )

    version_number = db.Column(db.Integer, nullable=False)  # 세션 내 단조 증가
    prompt_text = db.Column(db.Text, nullable=False)
    unified_diff = db.Column(db.Text)         # 이전 버전 대비 unified diff
    change_summary = db.Column(db.Text)       # 변경 사유 요약

    # 이 버전이 반영한 ClusterResult.cluster_key 목록 (JSON 배열)
    applied_cluster_keys_json = db.Column(db.Text, default="[]")

    # [팀 직접 구현] 컴포넌트 귀속 — 어떤 클러스터가 어떤 프롬프트 컴포넌트를 바꿨는지
    # JSON: [{"cluster_label": "...", "component": "subject|motion|camera|setting|style|audio|negative", "change": "..."}]
    component_attribution_json = db.Column(db.Text)

    # 롤백 지원: 이 버전으로 되돌린 경우
    rolled_back_at = db.Column(db.DateTime)
    rolled_back_from_version = db.Column(db.Integer)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    session = db.relationship("PipelineSession", backref="prompt_versions")
    video_round = db.relationship("VideoRound", backref="prompt_versions")

    def get_applied_cluster_keys(self) -> list[str]:
        return json.loads(self.applied_cluster_keys_json or "[]")

    def set_applied_cluster_keys(self, keys: list[str]):
        self.applied_cluster_keys_json = json.dumps(keys)

    def get_component_attribution(self) -> list[dict]:
        """[팀 직접 구현] 컴포넌트 귀속 데이터 반환."""
        return json.loads(self.component_attribution_json or "[]")

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "video_round_id": self.video_round_id,
            "version_number": self.version_number,
            "prompt_text": self.prompt_text,
            "unified_diff": self.unified_diff,
            "change_summary": self.change_summary,
            "applied_cluster_keys": self.get_applied_cluster_keys(),
            "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            "rolled_back_from_version": self.rolled_back_from_version,
            "created_at": self.created_at.isoformat(),
        }


class GlobalTechnicalSetting(db.Model):
    """
    [팀 직접 설계] 프로젝트 공통 기술 설정.

    scope='technical'인 클러스터(음성/자막/속도 등)는
    특정 세션이 아니라 전체 프로젝트에 반영할 수 있다.
    예: "자막 폰트 크기를 키워라" → 모든 세션의 자막 기본값 변경
    """
    __tablename__ = "global_technical_setting"

    id = db.Column(db.Integer, primary_key=True)

    # audio | subtitle | pacing | technical
    category = db.Column(db.String(50), nullable=False)
    setting_key = db.Column(db.String(100), nullable=False)   # 예: "subtitle_font_size"
    setting_value = db.Column(db.Text, nullable=False)        # 예: "18"
    description = db.Column(db.Text)

    # 이 설정을 유발한 클러스터 정보 (추적용)
    source_cluster_key = db.Column(db.String(36))
    source_session_id = db.Column(db.Integer, db.ForeignKey("pipeline_session.id"))
    source_round_id = db.Column(db.Integer, db.ForeignKey("video_round.id"))

    # 활성화 여부 (비활성화해도 기록은 남음)
    is_active = db.Column(db.Boolean, default=True)

    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("category", "setting_key", name="uq_global_setting"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "category": self.category,
            "setting_key": self.setting_key,
            "setting_value": self.setting_value,
            "description": self.description,
            "source_cluster_key": self.source_cluster_key,
            "source_session_id": self.source_session_id,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class PromptChange(db.Model):
    __tablename__ = "prompt_change"

    id = db.Column(db.Integer, primary_key=True)
    from_round_id = db.Column(
        db.Integer, db.ForeignKey("video_round.id"), nullable=False
    )
    to_round_id = db.Column(db.Integer, db.ForeignKey("video_round.id"))
    to_round = db.relationship(
        "VideoRound",
        foreign_keys=[to_round_id],
        uselist=False,
    )
    session_id = db.Column(
        db.Integer, db.ForeignKey("pipeline_session.id"), nullable=False
    )

    original_prompt = db.Column(db.Text, nullable=False)
    updated_prompt = db.Column(db.Text, nullable=False)
    change_summary = db.Column(db.Text)
    reflected_feedback_ids_json = db.Column(db.Text, default="[]")

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_reflected_feedback_ids(self):
        return json.loads(self.reflected_feedback_ids_json or "[]")

    def set_reflected_feedback_ids(self, ids):
        self.reflected_feedback_ids_json = json.dumps(ids)

    def to_dict(self):
        return {
            "id": self.id,
            "from_round_id": self.from_round_id,
            "to_round_id": self.to_round_id,
            "session_id": self.session_id,
            "original_prompt": self.original_prompt,
            "updated_prompt": self.updated_prompt,
            "change_summary": self.change_summary,
            "reflected_feedback_ids": self.get_reflected_feedback_ids(),
            "created_at": self.created_at.isoformat(),
        }


class Metric(db.Model):
    __tablename__ = "metric"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer, db.ForeignKey("pipeline_session.id"), nullable=False
    )
    round_number = db.Column(db.Integer, nullable=False)

    total_comments = db.Column(db.Integer, default=0)
    constructive_count = db.Column(db.Integer, default=0)
    spam_count = db.Column(db.Integer, default=0)
    inappropriate_count = db.Column(db.Integer, default=0)
    non_constructive_count = db.Column(db.Integer, default=0)
    reflected_count = db.Column(db.Integer, default=0)
    reflection_rate = db.Column(db.Float, default=0.0)

    auto_pipeline_seconds = db.Column(db.Float)
    manual_estimate_seconds = db.Column(db.Float)
    time_savings_percent = db.Column(db.Float)

    calculated_at = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "round_number": self.round_number,
            "total_comments": self.total_comments,
            "constructive_count": self.constructive_count,
            "spam_count": self.spam_count,
            "inappropriate_count": self.inappropriate_count,
            "non_constructive_count": self.non_constructive_count,
            "reflected_count": self.reflected_count,
            "reflection_rate": self.reflection_rate,
            "auto_pipeline_seconds": self.auto_pipeline_seconds,
            "manual_estimate_seconds": self.manual_estimate_seconds,
            "time_savings_percent": self.time_savings_percent,
            "calculated_at": self.calculated_at.isoformat(),
        }


class ShotRound(db.Model):
    """
    [팀 직접 구현] 샷 단위 피드백 루프 — 멀티샷 시퀀스의 개별 샷 정보 저장.

    finalize_shot_sequence 완료 시 생성됨.
    타임스탬프가 있는 댓글 → 해당 샷에 매핑 → 그 샷만 선택적 재생성 가능.
    """
    __tablename__ = "shot_round"

    id = db.Column(db.Integer, primary_key=True)
    video_round_id = db.Column(db.Integer, db.ForeignKey("video_round.id"), nullable=False)
    shot_number = db.Column(db.Integer, nullable=False)          # 1-based (1, 2, 3...)
    prompt = db.Column(db.Text)                                  # 이 샷의 생성 프롬프트
    video_file_path = db.Column(db.String(500))                  # 이 샷의 영상 파일 경로
    start_time_seconds = db.Column(db.Float, default=0.0)        # 최종 영상 내 시작 시간(초)
    end_time_seconds = db.Column(db.Float, default=8.0)          # 최종 영상 내 종료 시간(초)
    duration_seconds = db.Column(db.Float, default=8.0)
    # done | reused | generating | failed
    status = db.Column(db.String(20), default="done")
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    video_round = db.relationship("VideoRound", backref="shots")

    def to_dict(self):
        return {
            "id": self.id,
            "video_round_id": self.video_round_id,
            "shot_number": self.shot_number,
            "prompt": self.prompt,
            "video_file_path": self.video_file_path,
            "start_time_seconds": self.start_time_seconds,
            "end_time_seconds": self.end_time_seconds,
            "duration_seconds": self.duration_seconds,
            "status": self.status,
        }


class CharacterProfile(db.Model):
    """
    [팀 직접 구현] 캐릭터 일관성용 참조 이미지 프로파일 (힉스필드 Soul ID 대응).

    Veo image_url에 최대 3장 참조 이미지를 전달해 인물 외모·의상 일관성 유지.
    Soul ID처럼 파인튜닝 없이 즉시 사용 가능 (소프트 참조 방식).
    """
    __tablename__ = "character_profile"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer, db.ForeignKey("pipeline_session.id"), nullable=False
    )
    name = db.Column(db.String(100), nullable=False)
    reference_image_paths = db.Column(db.Text, default="[]")  # JSON 경로 목록 (최대 3)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_image_paths(self) -> list:
        try:
            return json.loads(self.reference_image_paths or "[]")[:3]
        except Exception:
            return []

    def to_image_url_string(self) -> str:
        """Veo image_url 파라미터용 쉼표 구분 경로 문자열."""
        return ",".join(self.get_image_paths())

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "name": self.name,
            "image_paths": self.get_image_paths(),
            "created_at": self.created_at.isoformat(),
        }

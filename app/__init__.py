import logging
import os
import threading
from flask import Flask
from .models import db

logger = logging.getLogger(__name__)


def _migrate_existing_tables(db_instance):
    """
    [팀 직접 구현] SQLite ALTER TABLE로 신규 컬럼 추가.
    db.create_all()은 기존 테이블의 신규 컬럼을 추가하지 않으므로 직접 처리.
    """
    new_cols = {
        "feedback": [
            ("timestamp_seconds", "FLOAT"),                         # 샷 단위 피드백: 댓글 타임스탬프
            ("shot_round_id", "INTEGER REFERENCES shot_round(id)"), # 샷 단위 피드백: 매핑된 샷
            ("specificity_score", "FLOAT"),
            ("validity_score", "FLOAT"),
            ("severity_score", "FLOAT"),
            ("urgency_score", "FLOAT"),
            ("quality_score", "FLOAT"),
            ("actionability_score", "FLOAT DEFAULT 0.5"),   # F-2: 실행가능성 축
            ("topic_category", "VARCHAR(50)"),
            ("topic_categories_json", "TEXT DEFAULT '[]'"),
            ("scope", "VARCHAR(20)"),
            ("detected_language", "VARCHAR(20) DEFAULT 'auto'"),
            ("cluster_key", "VARCHAR(36)"),
            ("needs_review", "BOOLEAN DEFAULT 0"),
        ],
        "cluster_result": [
            ("target_shot_number", "INTEGER"),                      # 샷 단위 피드백: 타깃 샷 번호
            ("avg_severity_score", "FLOAT"),
            ("avg_urgency_score", "FLOAT"),
            ("avg_quality_score", "FLOAT"),
            ("cohesion_score", "FLOAT"),
            ("cluster_summary", "TEXT"),
        ],
        "video_round": [
            ("cluster_results", None),                          # 관계 컬럼 — skip
            ("generator_adapter", "VARCHAR(20)"),
            ("estimated_cost_usd", "FLOAT DEFAULT 0.0"),
            ("music_file_path", "VARCHAR(500)"),
            ("music_adapter", "VARCHAR(30)"),
            ("music_prompt", "TEXT"),
            ("estimated_music_cost_usd", "FLOAT DEFAULT 0.0"),
            ("lipsync_file_path", "VARCHAR(500)"),
            ("lipsync_adapter", "VARCHAR(20)"),
            ("character_image_path", "VARCHAR(500)"),
            ("srt_file_path", "VARCHAR(500)"),
            ("youtube_metadata_json", "TEXT"),
            ("positive_feedback_ratio", "FLOAT DEFAULT 0.0"),   # F-1: 프롬프트 성능 메모리
            ("total_feedback_count", "INTEGER DEFAULT 0"),      # F-1: 프롬프트 성능 메모리
            ("last_frame_path", "VARCHAR(500)"),                # 프레임 체이닝: 다음 라운드 참조용
        ],
        "prompt_version": [
            ("component_attribution_json", "TEXT"),
        ],
        "pipeline_session": [
            ("canvas_json", "TEXT DEFAULT NULL"),   # Next.js 캔버스 노드/엣지 JSON
            ("owner_token", "VARCHAR(64)"),                     # H-4: 소유자 인증
            ("updated_at", "DATETIME"),             # 최종 수정 시각 (프론트엔드 Session.updated_at)
        ],
    }

    conn = db_instance.engine.raw_connection()
    try:
        cur = conn.cursor()
        for table, cols in new_cols.items():
            cur.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in cur.fetchall()}
            for col_name, col_type in cols:
                if col_type is None:
                    continue
                if col_name not in existing:
                    try:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                        logger.info("마이그레이션: %s.%s 컬럼 추가", table, col_name)
                    except Exception as e:
                        logger.warning("컬럼 추가 실패 (%s.%s): %s", table, col_name, e)
        conn.commit()
    finally:
        conn.close()


def _configure_sqlite(db_instance):
    """SQLite 성능 최적화 — WAL 모드, 캐시 크기, temp store."""
    try:
        from sqlalchemy import text
        with db_instance.engine.connect() as conn:
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.execute(text("PRAGMA synchronous=NORMAL"))
            conn.execute(text("PRAGMA cache_size=-32000"))   # 32MB 캐시
            conn.execute(text("PRAGMA temp_store=MEMORY"))
            conn.commit()
        logger.info("[Startup] SQLite WAL 모드 설정 완료")
    except Exception as e:
        logger.warning("[Startup] SQLite 최적화 설정 실패 (무시): %s", e)


def _reset_zombie_generating_rounds():
    """
    서버 재시작 시 generating 상태로 멈춘 라운드를 prompt_built로 복구.
    백그라운드 스레드는 서버 재시작과 함께 소멸하므로 해당 상태는 영구 좀비가 된다.
    """
    try:
        from .models import VideoRound, db as _db
        count = (
            _db.session.query(VideoRound)
            .filter(VideoRound.status == "generating")
            .update({"status": "prompt_built"})
        )
        if count:
            _db.session.commit()
            logger.warning("[Startup] generating 좀비 라운드 %d개를 prompt_built으로 복구", count)
    except Exception as e:
        logger.warning("[Startup] 좀비 라운드 복구 실패 (무시): %s", e)


def _warmup_models():
    """앱 시작 후 백그라운드에서 임베딩 모델 미리 로드 (첫 요청 지연 방지)."""
    try:
        from .pipeline.classifier.factory import create_classifier
        _ = create_classifier()
        logger.info("[Startup] 분류기 워밍업 완료")
    except Exception as e:
        logger.warning("[Startup] 분류기 워밍업 실패 (무시): %s", e)


def _fill_missing_owner_tokens():
    """데모 환경: owner_token 자동 생성 비활성화 (쿠키 유실로 인한 권한 오류 방지)."""
    pass


def create_app(config_object=None):
    app = Flask(__name__, instance_relative_config=True)

    if config_object is None:
        from config import Config
        app.config.from_object(Config)
    else:
        app.config.from_object(config_object)

    os.makedirs(app.instance_path, exist_ok=True)
    os.makedirs(app.config.get("UPLOAD_FOLDER", "uploads"), exist_ok=True)

    db.init_app(app)

    # CSRF 보호 초기화 (공유 인스턴스)
    from .extensions import limiter, csrf
    csrf.init_app(app)
    limiter.init_app(app)

    with app.app_context():
        db.create_all()
        _migrate_existing_tables(db)
        _configure_sqlite(db)
        _reset_zombie_generating_rounds()
        _fill_missing_owner_tokens()

    # 백그라운드 워밍업 (임베딩 모델 미리 로드)
    threading.Thread(target=_warmup_models, daemon=True).start()

    # Jinja2 전역 함수 등록
    app.jinja_env.globals.update(zip=zip, enumerate=enumerate, max=max, min=min)

    from .routes.main import main_bp
    from .routes.session import session_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(session_bp, url_prefix="/session")

    return app

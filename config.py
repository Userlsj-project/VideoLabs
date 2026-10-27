import os
import secrets
import warnings
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    _secret_key_raw = os.environ.get("SECRET_KEY", "")
    if not _secret_key_raw:
        warnings.warn(
            "SECRET_KEY 환경변수가 설정되지 않았습니다. "
            "프로덕션 배포 전에 반드시 강력한 랜덤 값으로 설정하세요.",
            stacklevel=2,
        )
        # 개발용: 재시작마다 랜덤 키 (세션 초기화됨)
        SECRET_KEY = secrets.token_hex(32)
    else:
        SECRET_KEY = _secret_key_raw

    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL", "sqlite:///pipeline.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # 세션 쿠키 보안 설정
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    # HTTPS 환경에서만 True로 설정 (로컬 개발 시 False)
    SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "false").lower() == "true"
    WTF_CSRF_TIME_LIMIT = 3600  # CSRF 토큰 1시간 유효

    # ── 프롬프트 생성 엔진 선택 ──────────────────────────────────────────────
    # auto: GEMINI_TEXT_API_KEY → ANTHROPIC_API_KEY → template 순으로 시도
    # gemini: GEMINI_TEXT_API_KEY 사용 (없으면 template 폴백)
    # claude: ANTHROPIC_API_KEY 사용 (없으면 template 폴백)
    # template: API 없이 규칙 기반 템플릿만 사용
    PROMPT_ENGINE = os.environ.get("PROMPT_ENGINE", "auto")

    # ── Gemini 텍스트 API (프롬프트 생성·업데이트 전용) ─────────────────────
    # ⚠️ 이 키에 결제를 연결하면 무료 티어가 깨짐.
    #    반드시 별도의 결제 미연결 Google Cloud 프로젝트에서 발급할 것.
    #    VEO_API_KEY와 절대 같은 프로젝트/키를 쓰지 말 것.
    GEMINI_TEXT_API_KEY = os.environ.get("GEMINI_TEXT_API_KEY")
    GEMINI_TEXT_MODEL = os.environ.get("GEMINI_TEXT_MODEL", "gemini-3.1-flash-lite")

    # ── Claude API (선택적 대체 옵션 — PROMPT_ENGINE=claude 시 사용) ─────────
    _anthropic_raw = os.environ.get("ANTHROPIC_API_KEY", "")
    ANTHROPIC_API_KEY = _anthropic_raw if _anthropic_raw.startswith("sk-ant-") else None
    CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")

    # ── YouTube Data API ─────────────────────────────────────────────────────
    _yt_raw = os.environ.get("YOUTUBE_API_KEY", "")
    YOUTUBE_API_KEY = _yt_raw if _yt_raw.startswith("AIza") else None
    YOUTUBE_CLIENT_SECRETS = os.environ.get("YOUTUBE_CLIENT_SECRETS_FILE", "client_secrets.json")
    YOUTUBE_CREDENTIALS = os.environ.get("YOUTUBE_CREDENTIALS_FILE", "token.json")

    # ── AI 영상 생성 어댑터 ──────────────────────────────────────────────────
    # 기본 권장: veo (대사 있는 장면·실제 제출용 최종 영상)
    VIDEO_GENERATOR_ADAPTER = os.environ.get("VIDEO_GENERATOR_ADAPTER", "mock")

    # ── Veo 3.1 API (영상 생성용, 과금 발생) ─────────────────────────────────
    # ⚠️ GEMINI_TEXT_API_KEY와 절대 같은 프로젝트/키를 쓰지 말 것.
    #    Veo API 호출은 과금 발생 — 텍스트 생성용 무료 티어와 분리 필수.
    VEO_API_KEY = os.environ.get("VEO_API_KEY")
    VEO_QUALITY_TIER = os.environ.get("VEO_QUALITY_TIER", "fast")   # fast|standard|high
    VEO_MODEL = os.environ.get("VEO_MODEL", "veo-3.1-generate-preview")
    # 8 = 단일 클립(기본). 30 = 베이스 8초 + Extend 4회 = 36초.
    # 60 = 베이스 8초 + Extend 8회 = 64초.
    VEO_EXTEND_TARGET_SECONDS = int(os.environ.get("VEO_EXTEND_TARGET_SECONDS", "8"))

    RUNWAY_API_KEY = os.environ.get("RUNWAY_API_KEY")

    # ── fal.ai Seedance API (영상 생성용, 과금 발생) ──────────────────────────
    # fal.ai: ByteDance Seedance 등 다양한 영상 생성 모델 제공 플랫폼
    # FAL_MODEL: seedance-1-lite (빠름·저렴) | seedance-1-pro (고품질)
    FAL_API_KEY = os.environ.get("FAL_API_KEY")
    FAL_MODEL = os.environ.get("FAL_MODEL", "fal-ai/bytedance/seedance-1-lite")

    # ── fal.ai Kling V3 Standard (영상 생성용, 과금 발생) ────────────────────
    # FAL_API_KEY를 공유 사용 (Kling도 fal.ai 플랫폼)
    # $0.084/초 (오디오 OFF 기준), 최대 15초/클립, 1080p. 15초 초과 시 자동 분할 + ffmpeg concat.
    # ⚠️ 모델명 변경 주의: standard → pro 로 바꾸면 단가 $0.112/초로 상승
    KLING_MODEL = os.environ.get("KLING_MODEL", "fal-ai/kling-video/v3/standard/text-to-video")
    KLING_EXTEND_TARGET_SECONDS = int(os.environ.get("KLING_EXTEND_TARGET_SECONDS", "22"))

    # ── 유료 호출 일일 안전장치 ──────────────────────────────────────────────
    MAX_PAID_GENERATIONS_PER_DAY = int(
        os.environ.get("MAX_PAID_GENERATIONS_PER_DAY", "20")
    )

    # ── 시뮬레이션 모드 ──────────────────────────────────────────────────────
    SIMULATION_MODE = os.environ.get("SIMULATION_MODE", "true").lower() == "true"
    MANUAL_TIME_PER_FEEDBACK_MINUTES = float(
        os.environ.get("MANUAL_TIME_PER_FEEDBACK_MINUTES", "3")
    )

    # ── [분류기] 임베딩 기반 다국어 분류기 (인증 불필요) ─────────────────────
    # 모델: paraphrase-multilingual-MiniLM-L12-v2 (클러스터링 모델과 공유)
    # 라이선스: Apache 2.0  |  ~118MB  |  CPU  |  50+ 언어  |  계정 불필요
    # 분류 방식: 원형 문장 코사인 유사도 + 구체성·타당성 규칙 패턴 2축 판단
    CLASSIFIER_MODEL = os.environ.get(
        "CLASSIFIER_MODEL",
        "paraphrase-multilingual-MiniLM-L12-v2",
    )
    # 두 축 판별 임계값 (factory에서 EmbeddingClassifier용으로 자동 조정됨)
    SPECIFICITY_THRESHOLD = float(os.environ.get("SPECIFICITY_THRESHOLD", "0.45"))
    VALIDITY_THRESHOLD = float(os.environ.get("VALIDITY_THRESHOLD", "0.40"))

    # ── [임베딩] 로컬 다국어 문장 임베딩 모델 ────────────────────────────────
    # 모델: paraphrase-multilingual-mpnet-base-v2
    # 라이선스: Apache 2.0  |  파라미터: ~278M  |  50+ 언어
    # 폴백: paraphrase-multilingual-MiniLM-L12-v2 (경량, 이미 캐시됨)
    EMBEDDING_MODEL = os.environ.get(
        "EMBEDDING_MODEL",
        "paraphrase-multilingual-mpnet-base-v2",
    )
    EMBEDDING_FALLBACK_MODEL = os.environ.get(
        "EMBEDDING_FALLBACK_MODEL",
        "paraphrase-multilingual-MiniLM-L12-v2",
    )

    # ── [클러스터링] 계층적 군집화 파라미터 ──────────────────────────────────
    # distance_threshold: 코사인 거리 기본값 (카테고리별로 자동 오버라이드됨)
    CLUSTERING_DISTANCE_THRESHOLD = float(
        os.environ.get("CLUSTERING_DISTANCE_THRESHOLD", "0.40")
    )
    # 우선순위 점수 가중치 (v2: 5축 공식)
    # P = α·빈도 + β·품질 + γ·심각도 + δ·긴급도 + ε·타당성
    PRIORITY_ALPHA = float(os.environ.get("PRIORITY_ALPHA", "0.35"))    # 빈도
    PRIORITY_BETA = float(os.environ.get("PRIORITY_BETA", "0.25"))      # 품질
    PRIORITY_GAMMA = float(os.environ.get("PRIORITY_GAMMA", "0.20"))    # 심각도
    PRIORITY_DELTA = float(os.environ.get("PRIORITY_DELTA", "0.12"))    # 긴급도
    PRIORITY_EPSILON = float(os.environ.get("PRIORITY_EPSILON", "0.08"))  # 타당성

    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", "uploads")
    MAX_CONTENT_LENGTH = 500 * 1024 * 1024

    # ── 음악 생성 어댑터 (Phase 2) ────────────────────────────────────────────
    # elevenlabs_music: vocals+BGM 동시 생성 (유료 Starter $5/월 필요)
    # elevenlabs_sound: BGM only (무료 티어 가능)
    # mock: 개발·테스트용 (기본값)
    MUSIC_GENERATOR_ADAPTER = os.environ.get("MUSIC_GENERATOR_ADAPTER", "mock")
    ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
    # 나레이션 TTS 전용 보이스 ID (ElevenLabs pre-made voice, 무료 플랜 사용 가능)
    # 기본: Adam(pNInz6obpgDQGcFmaJgB). 변경 시 ElevenLabs 보이스 라이브러리에서 ID 확인
    ELEVENLABS_NARRATION_VOICE_ID = os.environ.get("ELEVENLABS_NARRATION_VOICE_ID", "pNInz6obpgDQGcFmaJgB")
    MAX_MUSIC_GENERATIONS_PER_DAY = int(
        os.environ.get("MAX_MUSIC_GENERATIONS_PER_DAY", "10")
    )

    # ── 립싱크 어댑터 (Phase 3) ───────────────────────────────────────────────
    # sadtalker: 표정+입 자연스러움, 비상업 연구용 라이선스 → 졸작 OK
    # wav2lip: 입만 교체, MIT 라이선스 → sadtalker 폴백
    # mock: 개발·테스트용 (기본값)
    LIPSYNC_ADAPTER = os.environ.get("LIPSYNC_ADAPTER", "mock")
    SADTALKER_DIR = os.environ.get("SADTALKER_DIR", "SadTalker")
    WAV2LIP_DIR = os.environ.get("WAV2LIP_DIR", "Wav2Lip")
    DEFAULT_CHARACTER_IMAGE = os.environ.get("DEFAULT_CHARACTER_IMAGE", "")

"""공유 Flask 확장 인스턴스 — 순환 임포트 방지를 위해 별도 모듈로 분리."""
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect

# 앱 없이 생성, create_app()에서 init_app()으로 연결
# storage_uri="memory://"로 in-memory 스토리지 명시 (개발/단일 워커 환경)
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    storage_uri="memory://",
)

csrf = CSRFProtect()

"""
[팀 직접 구현] API 키·민감 정보 마스킹 유틸리티.

외부 API(Veo, Runway, Gemini, YouTube) 호출 시 예외 로그에
키·헤더가 노출되지 않도록 공통으로 사용한다.
"""

from __future__ import annotations

import re


def mask_key(key: str | None, visible: int = 6) -> str:
    """API 키 중간 부분을 마스킹하여 반환. 로그 출력 전용."""
    if not key:
        return "(없음)"
    if len(key) <= visible * 2:
        return "*" * len(key)
    return key[:visible] + "***" + key[-visible:]


def sanitize_headers(headers: dict) -> dict:
    """Authorization, X-API-Key 등 민감 헤더를 마스킹한 복사본 반환."""
    sensitive = {"authorization", "x-api-key", "x-goog-api-key", "x-runway-api-key"}
    return {
        k: (mask_key(v) if k.lower() in sensitive else v)
        for k, v in headers.items()
    }


# key=value / key: value 형식
_KEY_PATTERN = re.compile(
    r"(key|token|secret|password)\s*[=:]\s*\S+",
    re.IGNORECASE,
)
# Authorization: Bearer <token> — 공백 구분이라 별도 패턴 필요
_BEARER_PATTERN = re.compile(
    r"bearer\s+[A-Za-z0-9._\-/+]{8,}",
    re.IGNORECASE,
)


def sanitize_exception_msg(exc: Exception) -> str:
    """예외 메시지에서 API 키 패턴을 마스킹하여 반환."""
    msg = str(exc)
    msg = _KEY_PATTERN.sub(lambda m: m.group().split("=")[0].split(":")[0].strip() + " [REDACTED]", msg)
    msg = _BEARER_PATTERN.sub("bearer [REDACTED]", msg)
    return msg

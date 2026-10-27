"""
Google Gemini API Veo 3.1 어댑터.

VEO_API_KEY 환경변수 설정 후 사용 가능.
이 키는 반드시 Veo 사용이 활성화된 Google Cloud 프로젝트의 키여야 한다.
⚠️ GEMINI_TEXT_API_KEY와 절대 같은 키/프로젝트를 쓰지 말 것
   (Veo API는 과금 발생 — 텍스트 생성 무료 티어가 깨진다).

생성 흐름 (기본, extend_target_seconds=8):
  1. _validate_prompt()       — 빈 프롬프트·길이 프리플라이트 검증
  2. _submit_with_retry()     — API 제출 (일시적 오류: 지수 백오프 재시도,
                                 콘텐츠 정책 위반: 즉시 실패 + 사유 기록)
  3. _poll_until_done()       — Operation 완료 폴링 (최대 15분)
  4. _download()              — 완성 영상 파일 저장

Extend 흐름 (extend_target_seconds > 8):
  위 기본 흐름으로 8초 베이스 클립 생성 후,
  _extend_to_duration()으로 7초씩 최대 20회 연장.
  각 연장은 직전 클립 URI를 참조해 장면·스타일·모션 연속성 유지.
  최종 다운로드된 영상 하나만 저장 (중간 파일 자동 삭제).

화질 등급: VEO_QUALITY_TIER 환경변수로 설정 (기본값: fast).
참조 이미지: image_url 있을 때만 포함 (Ingredients to Video, 최대 3장).
"""

from __future__ import annotations

import logging
import math
import os
import time

from .base import BaseVideoGeneratorAdapter, VideoGenerationResult
from ..utils.masking import mask_key, sanitize_exception_msg

logger = logging.getLogger(__name__)

_PROMPT_MAX_LEN = 1500
_BANNED_KEYWORDS = frozenset({"nsfw", "nude", "explicit", "porn", "graphic violence"})
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}

_BASE_DURATION_SECONDS = 8    # Veo 기본 클립 길이
_EXTEND_STEP_SECONDS = 7      # 연장 1회당 추가 길이
_MAX_EXTENSIONS = 20          # 최대 연장 횟수 (API 제한)

# 콘텐츠 정책 위반 감지 키워드 (HTTP 400 전체가 아닌 정책 위반 메시지만 대상)
_POLICY_ERROR_KEYWORDS = frozenset({"policy", "safety", "blocked", "prohibited", "violat", "inappropriate"})

# VEO_QUALITY_TIER → Veo generate_videos_config quality 매핑
_QUALITY_MAP = {
    "fast": "QUALITY_FAST",
    "standard": "QUALITY_STANDARD",
    "high": "QUALITY_HIGH",
}


class VeoAdapter(BaseVideoGeneratorAdapter):
    """
    Google Veo 3.1 어댑터.
    google-genai 패키지(>=0.8.0) 필요: pip install google-genai
    """

    def __init__(
        self,
        api_key: str,
        quality_tier: str = "fast",
        upload_folder: str = "uploads",
        model: str = "veo-3.1-generate-preview",
        extend_target_seconds: int = 8,
    ):
        self.api_key = api_key
        self.quality_tier = quality_tier
        self.upload_folder = upload_folder
        self.model = model
        # 8이하면 단순 단일 생성, 9이상이면 Extend 루프 사용
        self.extend_target_seconds = max(extend_target_seconds, _BASE_DURATION_SECONDS)
        os.makedirs(upload_folder, exist_ok=True)

    def adapter_name(self) -> str:
        return "veo"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        genre: str = "",
        image_url: str | None = None,
    ) -> VideoGenerationResult:
        # 1. 프리플라이트 검증
        ok, reason = self._validate_prompt(prompt)
        if not ok:
            return VideoGenerationResult(success=False, error_message=f"프롬프트 검증 실패: {reason}")

        # 2. API 제출 (재시도 포함)
        try:
            operation = self._submit_with_retry(prompt, image_url)
        except _ContentPolicyError as e:
            logger.warning("[Veo] 콘텐츠 정책 위반 거부 (key=%s): %s", mask_key(self.api_key), e)
            return VideoGenerationResult(success=False, error_message=f"콘텐츠 정책 위반: {e}")
        except Exception as e:
            logger.error("[Veo] API 요청 실패 (key=%s): %s", mask_key(self.api_key), sanitize_exception_msg(e))
            return VideoGenerationResult(success=False, error_message=f"API 요청 실패: {sanitize_exception_msg(e)}")

        # 3. 완료 폴링 (베이스 클립)
        base_result = self._poll_until_done(operation)
        if not base_result.success:
            return base_result

        # 4. Extend: 목표 시간이 기본 8초를 초과하면 반복 연장
        if self.extend_target_seconds > _BASE_DURATION_SECONDS:
            return self._extend_to_duration(prompt, base_result)

        return base_result

    # ── 이미지 참조 처리 ─────────────────────────────────────────────────────

    def _resolve_images(self, image_url: str) -> list:
        """
        쉼표 구분 URL/로컬경로 문자열을 gtypes.Image 목록으로 변환 (최대 3장).

        - http(s)://로 시작하면 gcs_uri로 전달 (공개 URL 직접 참조)
        - 로컬 파일 경로이면 Image.from_file()로 바이트 로드 (base64 인라인)
        """
        try:
            from google.genai import types as gtypes
        except ImportError:
            return []

        images = []
        for raw in image_url.split(","):
            url_or_path = raw.strip()
            if not url_or_path:
                continue
            try:
                if url_or_path.startswith(("http://", "https://")):
                    images.append(gtypes.Image(gcs_uri=url_or_path))
                elif os.path.exists(url_or_path):
                    images.append(gtypes.Image.from_file(location=url_or_path))
                else:
                    logger.warning("[Veo] image_url 경로 없음, 건너뜀: %s", url_or_path)
            except Exception as e:
                logger.warning("[Veo] 이미지 로드 실패 (%s): %s", url_or_path, e)
            if len(images) >= 3:
                break
        return images

    # ── 검증 ────────────────────────────────────────────────────────────────

    def _validate_prompt(self, prompt: str) -> tuple[bool, str]:
        if not prompt.strip():
            return False, "프롬프트가 비어 있습니다"
        if len(prompt) > _PROMPT_MAX_LEN:
            return False, f"프롬프트 길이 초과 ({len(prompt)}자 > {_PROMPT_MAX_LEN}자)"
        lower = prompt.lower()
        for kw in _BANNED_KEYWORDS:
            if kw in lower:
                return False, f"금지 키워드 포함: '{kw}'"
        return True, ""

    # ── API 제출 (지수 백오프 재시도) ────────────────────────────────────────

    def _submit_with_retry(
        self,
        prompt: str,
        image_url: str | None,
        max_retries: int = 3,
    ):
        """생성 작업을 제출하고 GenerateVideosOperation 객체를 반환한다."""
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError as e:
            raise RuntimeError(
                "google-genai 패키지가 설치되지 않았습니다. "
                "pip install google-genai 를 실행하세요."
            ) from e

        client = genai.Client(api_key=self.api_key)

        config = gtypes.GenerateVideosConfig(
            aspect_ratio="16:9",
            number_of_videos=1,
            duration_seconds=8,
        )

        last_exc: Exception = RuntimeError("알 수 없는 오류")
        for attempt in range(max_retries):
            try:
                if image_url:
                    # Ingredients to Video: 참조 이미지 포함 (HTTP URL 또는 로컬 경로)
                    images = self._resolve_images(image_url)
                    if images:
                        operation = client.models.generate_videos(
                            model=self.model,
                            prompt=prompt,
                            image=images[0] if len(images) == 1 else images,
                            config=config,
                        )
                    else:
                        operation = client.models.generate_videos(
                            model=self.model,
                            prompt=prompt,
                            config=config,
                        )
                else:
                    operation = client.models.generate_videos(
                        model=self.model,
                        prompt=prompt,
                        config=config,
                    )
                return operation  # GenerateVideosOperation 객체 반환

            except Exception as e:
                last_exc = e
                err_msg = str(e).lower()
                # 콘텐츠 정책 위반: 재시도 없이 즉시 실패
                if any(kw in err_msg for kw in _POLICY_ERROR_KEYWORDS):
                    raise _ContentPolicyError(str(e)) from e
                # 일시적 오류(429/5xx)에만 재시도 — 그 외는 즉시 실패 (중복 과금 방지)
                is_transient = any(str(code) in err_msg for code in _RETRYABLE_HTTP_CODES)
                if not is_transient or attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning("[Veo] 제출 실패 (시도 %d/%d), %ds 후 재시도: %s",
                               attempt + 1, max_retries, wait, sanitize_exception_msg(e))
                time.sleep(wait)

        raise last_exc

    # ── 완료 폴링 ────────────────────────────────────────────────────────────

    def _poll_until_done(
        self,
        operation,
        max_polls: int = 90,
        interval: int = 10,
    ) -> VideoGenerationResult:
        """GenerateVideosOperation 객체를 받아 최대 max_polls × interval초(기본 15분) 동안 폴링한다."""
        try:
            from google import genai
        except ImportError as e:
            return VideoGenerationResult(success=False, error_message=str(e))

        client = genai.Client(api_key=self.api_key)
        operation_name = getattr(operation, "name", str(operation))

        for poll_num in range(max_polls):
            time.sleep(interval)
            try:
                # SDK는 operation 객체를 그대로 전달받아 최신 상태를 반환한다
                operation = client.operations.get(operation)
            except Exception as e:
                return VideoGenerationResult(
                    success=False,
                    error_message=f"폴링 오류: {sanitize_exception_msg(e)}",
                )

            if not operation.done:
                logger.debug("[Veo] 폴링 %d/%d — 아직 생성 중", poll_num + 1, max_polls)
                continue

            # 오류 확인
            if hasattr(operation, "error") and operation.error and operation.error.message:
                return VideoGenerationResult(
                    success=False,
                    error_message=f"생성 실패: {operation.error.message}",
                )

            # 결과 추출 (response 속성 사용 — result 아님)
            try:
                response = operation.response
                videos = response.generated_videos if hasattr(response, "generated_videos") else []
                if not videos:
                    return VideoGenerationResult(success=False, error_message="생성된 영상 없음")

                video_uri = videos[0].video.uri
                file_path = self._download(client, video_uri, operation_name)
                return VideoGenerationResult(
                    success=True,
                    file_path=file_path,
                    metadata={
                        "operation_name": operation_name,
                        "adapter": "veo",
                        "model": self.model,
                        "quality_tier": self.quality_tier,
                        "video_uri": video_uri,   # Extend 연결에 사용
                        "duration_seconds": _BASE_DURATION_SECONDS,
                    },
                )
            except Exception as e:
                return VideoGenerationResult(
                    success=False,
                    error_message=f"결과 처리 오류: {sanitize_exception_msg(e)}",
                )

        return VideoGenerationResult(success=False, error_message="생성 시간 초과 (15분)")

    # ── 다운로드 ─────────────────────────────────────────────────────────────

    def _download(self, client, uri: str, operation_name: str) -> str:
        """생성된 영상을 로컬에 저장하고 경로를 반환한다."""
        import requests
        safe_name = operation_name.replace("/", "_")
        file_path = os.path.join(self.upload_folder, f"veo_{safe_name}.mp4")

        try:
            # google-genai SDK 방식으로 다운로드 시도
            file_data = client.files.download(file=uri)
            with open(file_path, "wb") as f:
                f.write(file_data)
        except Exception:
            # API 키 인증이 필요한 URI (generativelanguage.googleapis.com)
            resp = requests.get(uri, params={"key": self.api_key}, timeout=120)
            if resp.status_code == 403:
                # 직접 URL인 경우 파라미터 없이 재시도
                resp = requests.get(uri, timeout=120)
            resp.raise_for_status()
            with open(file_path, "wb") as f:
                f.write(resp.content)

        return file_path


    # ── Extend: 7초씩 연장 ───────────────────────────────────────────────────

    def _extend_to_duration(
        self,
        prompt: str,
        base_result: VideoGenerationResult,
    ) -> VideoGenerationResult:
        """베이스 클립을 extend_target_seconds까지 7초씩 반복 연장한다.

        연장에 실패하면 중단하고 그 시점까지의 결과를 반환 (graceful degradation).
        연장 성공 시 중간 다운로드 파일은 자동 삭제해 디스크 절약.
        """
        extensions_needed = min(
            math.ceil((self.extend_target_seconds - _BASE_DURATION_SECONDS) / _EXTEND_STEP_SECONDS),
            _MAX_EXTENSIONS,
        )
        logger.info(
            "[Veo-Extend] 목표 %d초 → 연장 %d회 예정 (8 + %d×7 = %d초)",
            self.extend_target_seconds,
            extensions_needed,
            extensions_needed,
            _BASE_DURATION_SECONDS + extensions_needed * _EXTEND_STEP_SECONDS,
        )

        current_result = base_result
        current_duration = _BASE_DURATION_SECONDS

        for i in range(extensions_needed):
            source_uri = (current_result.metadata or {}).get("video_uri")
            if not source_uri:
                logger.warning("[Veo-Extend] video_uri 없음 — 연장 중단 (%d초에서 반환)", current_duration)
                break

            logger.info("[Veo-Extend] 연장 %d/%d (현재 %d초)", i + 1, extensions_needed, current_duration)

            try:
                operation = self._submit_extend(prompt, source_uri)
            except _ContentPolicyError as e:
                logger.warning("[Veo-Extend] 콘텐츠 정책 위반 — 현재 영상 반환: %s", e)
                break
            except Exception as e:
                logger.warning("[Veo-Extend] 제출 실패 — 현재 영상 반환: %s", sanitize_exception_msg(e))
                break

            ext_result = self._poll_until_done(operation)
            if not ext_result.success:
                logger.warning("[Veo-Extend] 폴링 실패 — 현재 영상 반환: %s", ext_result.error_message)
                break

            # 중간 파일 삭제 (마지막 결과물만 유지)
            prev_path = current_result.file_path
            current_duration += _EXTEND_STEP_SECONDS
            if ext_result.metadata:
                ext_result.metadata["duration_seconds"] = current_duration
            current_result = ext_result
            if prev_path and os.path.exists(prev_path):
                try:
                    os.remove(prev_path)
                except OSError:
                    pass

        logger.info("[Veo-Extend] 완료 — 최종 %d초", current_duration)
        return current_result

    def _submit_extend(
        self,
        prompt: str,
        source_video_uri: str,
        max_retries: int = 3,
    ):
        """기존 영상 URI를 참조해 7초 연장 작업을 제출하고 GenerateVideosOperation 객체를 반환한다."""
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError as e:
            raise RuntimeError("google-genai 패키지 미설치: pip install google-genai") from e

        client = genai.Client(api_key=self.api_key)

        config = gtypes.GenerateVideosConfig(
            aspect_ratio="16:9",
            number_of_videos=1,
            duration_seconds=_EXTEND_STEP_SECONDS,
        )

        last_exc: Exception = RuntimeError("알 수 없는 오류")
        for attempt in range(max_retries):
            try:
                operation = client.models.generate_videos(
                    model=self.model,
                    prompt=prompt,
                    video=gtypes.Video(uri=source_video_uri),
                    config=config,
                )
                return operation  # GenerateVideosOperation 객체 반환
            except Exception as e:
                last_exc = e
                err_msg = str(e).lower()
                if any(kw in err_msg for kw in _POLICY_ERROR_KEYWORDS):
                    raise _ContentPolicyError(str(e)) from e
                is_transient = any(str(code) in err_msg for code in _RETRYABLE_HTTP_CODES)
                if not is_transient or attempt == max_retries - 1:
                    raise
                wait = 2 ** attempt
                logger.warning(
                    "[Veo-Extend] 제출 실패 (시도 %d/%d), %ds 후 재시도: %s",
                    attempt + 1, max_retries, wait, sanitize_exception_msg(e),
                )
                time.sleep(wait)

        raise last_exc


class _ContentPolicyError(Exception):
    """콘텐츠 정책 위반 — 재시도 없이 즉시 실패."""

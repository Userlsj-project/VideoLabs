"""
fal.ai Seedance 어댑터.

FAL_API_KEY 환경변수 설정 후 사용 가능.
fal.ai는 ByteDance Seedance 등 다양한 영상 생성 모델을 API로 제공하는 플랫폼이다.

생성 흐름:
  1. _validate_prompt() — 길이·금지어 프리플라이트 검증
  2. fal_client.subscribe() — API 제출 + 완료 대기 (내부 폴링 포함)
  3. _download() — 완성 영상 파일 저장
"""

from __future__ import annotations

import logging
import os

import requests

from .base import BaseVideoGeneratorAdapter, VideoGenerationResult
from ..utils.masking import mask_key, sanitize_exception_msg

logger = logging.getLogger(__name__)

_PROMPT_MAX_LEN = 2000
_BANNED_KEYWORDS = frozenset({"nsfw", "nude", "explicit", "porn", "graphic violence"})

# fal.ai Seedance 모델 ID (lite: 빠름·저렴, pro: 고품질)
DEFAULT_MODEL = "fal-ai/bytedance/seedance-1-lite"


class FalAdapter(BaseVideoGeneratorAdapter):
    """
    [팀 직접 구현] fal.ai Seedance 영상 생성 어댑터.
    FAL_API_KEY + fal-client 패키지 필요.
    """

    def __init__(
        self,
        api_key: str,
        upload_folder: str = "uploads",
        model: str = DEFAULT_MODEL,
        duration: int = 5,
        resolution: str = "720p",
        aspect_ratio: str = "16:9",
    ):
        self.api_key = api_key
        self.upload_folder = upload_folder
        self.model = model
        self.duration = duration
        self.resolution = resolution
        self.aspect_ratio = aspect_ratio
        os.makedirs(upload_folder, exist_ok=True)

    def adapter_name(self) -> str:
        return "fal"

    def is_available(self) -> bool:
        return bool(self.api_key)

    def generate(
        self,
        prompt: str,
        genre: str = "",
        image_url: str | None = None,
    ) -> VideoGenerationResult:
        ok, reason = self._validate_prompt(prompt)
        if not ok:
            return VideoGenerationResult(success=False, error_message=f"프롬프트 검증 실패: {reason}")

        try:
            import fal_client  # noqa: PLC0415
        except ImportError:
            return VideoGenerationResult(
                success=False,
                error_message="fal-client 패키지가 설치되지 않았습니다. pip install fal-client",
            )

        # fal_client는 환경변수 FAL_KEY로 인증
        os.environ["FAL_KEY"] = self.api_key

        arguments: dict = {
            "prompt": prompt[:_PROMPT_MAX_LEN],
            "duration": self.duration,
            "resolution": self.resolution,
            "aspect_ratio": self.aspect_ratio,
        }
        if image_url:
            arguments["image_url"] = image_url.split(",")[0].strip()  # 첫 번째 이미지만 사용

        try:
            logger.info("[Fal] 영상 생성 요청 (model=%s, key=%s)", self.model, mask_key(self.api_key))
            result = fal_client.subscribe(self.model, arguments=arguments)
        except Exception as e:
            logger.error("[Fal] API 요청 실패 (key=%s): %s", mask_key(self.api_key), sanitize_exception_msg(e))
            return VideoGenerationResult(success=False, error_message=f"API 요청 실패: {sanitize_exception_msg(e)}")

        try:
            video_url = result["video"]["url"]
            request_id = result.get("request_id", "unknown")
        except (KeyError, TypeError) as e:
            return VideoGenerationResult(success=False, error_message=f"응답 파싱 실패: {e}")

        try:
            file_path = self._download(video_url, request_id)
        except Exception as e:
            return VideoGenerationResult(success=False, error_message=f"다운로드 실패: {sanitize_exception_msg(e)}")

        logger.info("[Fal] 영상 생성 완료 → %s", file_path)
        return VideoGenerationResult(
            success=True,
            file_path=file_path,
            metadata={"model": self.model, "adapter": "fal", "request_id": request_id},
        )

    # ── 검증 ─────────────────────────────────────────────────────────────────

    def _validate_prompt(self, prompt: str) -> tuple[bool, str]:
        if not prompt.strip():
            return False, "프롬프트가 비어 있습니다"
        lower = prompt.lower()
        for kw in _BANNED_KEYWORDS:
            if kw in lower:
                return False, f"금지 키워드 포함: '{kw}'"
        return True, ""

    # ── 다운로드 ──────────────────────────────────────────────────────────────

    def _download(self, url: str, request_id: str) -> str:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        file_path = os.path.join(self.upload_folder, f"fal_{request_id}.mp4")
        with open(file_path, "wb") as f:
            f.write(resp.content)
        return file_path

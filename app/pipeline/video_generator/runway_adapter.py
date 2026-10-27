"""
RunwayML Gen-3 어댑터.
RUNWAY_API_KEY 환경변수 설정 후 사용 가능.

생성 흐름:
  1. _validate_prompt()  — 길이·금지어 프리플라이트 검증
  2. _submit_with_retry() — API 제출 (일시적 오류 시 지수 백오프 재시도)
  3. _poll_until_done()  — 생성 완료 폴링 (최대 10분)
  4. _download()        — 완성 영상 파일 저장
"""

import logging
import os
import time
import requests

from .base import BaseVideoGeneratorAdapter, VideoGenerationResult
from ..utils.masking import mask_key, sanitize_exception_msg

logger = logging.getLogger(__name__)

_PROMPT_MAX_LEN = 1000
_BANNED_KEYWORDS = frozenset({"nsfw", "nude", "explicit", "porn", "graphic violence"})
_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}


class RunwayAdapter(BaseVideoGeneratorAdapter):
    API_BASE = "https://api.dev.runwayml.com/v1"

    def __init__(self, api_key: str, upload_folder: str = "uploads"):
        self.api_key = api_key
        self.upload_folder = upload_folder
        os.makedirs(upload_folder, exist_ok=True)

    def adapter_name(self) -> str:
        return "runway"

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
            task_id = self._submit_with_retry(prompt, image_url)
        except Exception as e:
            logger.error("[Runway] API 요청 실패 (key=%s): %s", mask_key(self.api_key), sanitize_exception_msg(e))
            return VideoGenerationResult(success=False, error_message=f"API 요청 실패: {sanitize_exception_msg(e)}")

        # 3. 완료 폴링
        return self._poll_until_done(task_id)

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
    ) -> str:
        """생성 작업을 제출하고 task_id를 반환한다."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload: dict = {
            "promptText": prompt,
            "model": "gen3a_turbo",
            "duration": 10,
            "ratio": "1280:768",
        }
        if image_url:
            # image_url이 있으면 첫 프레임으로 사용 (image-to-video 모드)
            payload["promptImage"] = image_url

        last_exc: Exception = RuntimeError("알 수 없는 오류")
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{self.API_BASE}/image_to_video",
                    headers=headers,
                    json=payload,
                    timeout=30,
                )
                resp.raise_for_status()
                return resp.json()["id"]
            except requests.exceptions.HTTPError as e:
                last_exc = e
                code = e.response.status_code if e.response is not None else 0
                if code not in _RETRYABLE_HTTP_CODES or attempt == max_retries - 1:
                    raise
            except requests.exceptions.ConnectionError as e:
                last_exc = e
                if attempt == max_retries - 1:
                    raise
            # 지수 백오프: 1s → 2s → 4s
            time.sleep(2 ** attempt)

        raise last_exc

    # ── 완료 폴링 ────────────────────────────────────────────────────────────

    def _poll_until_done(
        self,
        task_id: str,
        max_polls: int = 60,
        interval: int = 10,
    ) -> VideoGenerationResult:
        """최대 max_polls × interval초(기본 10분) 동안 폴링한다."""
        headers = {"Authorization": f"Bearer {self.api_key}"}
        for _ in range(max_polls):
            time.sleep(interval)
            try:
                resp = requests.get(
                    f"{self.API_BASE}/tasks/{task_id}",
                    headers=headers,
                    timeout=10,
                )
                resp.raise_for_status()
                task = resp.json()
            except Exception as e:
                return VideoGenerationResult(success=False, error_message=f"폴링 오류: {sanitize_exception_msg(e)}")

            status = task.get("status", "")
            if status == "SUCCEEDED":
                video_url = task["output"][0]
                file_path = self._download(video_url, task_id)
                return VideoGenerationResult(
                    success=True,
                    file_path=file_path,
                    metadata={"task_id": task_id, "adapter": "runway"},
                )
            if status == "FAILED":
                reason = task.get("failure") or task.get("failureCode", "알 수 없는 실패")
                return VideoGenerationResult(success=False, error_message=f"생성 실패: {reason}")

        return VideoGenerationResult(success=False, error_message="생성 시간 초과 (10분)")

    # ── 다운로드 ─────────────────────────────────────────────────────────────

    def _download(self, url: str, task_id: str) -> str:
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        file_path = os.path.join(self.upload_folder, f"runway_{task_id}.mp4")
        with open(file_path, "wb") as f:
            f.write(resp.content)
        return file_path

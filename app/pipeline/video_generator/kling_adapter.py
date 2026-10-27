"""
fal.ai Kling V3 Standard 어댑터.

FAL_API_KEY 환경변수 설정 후 사용 가능.
Kling V3 Standard: 최대 15초/클립, 1080p, $0.084/초 (오디오 비활성화 기준)

22초 이상 생성 시 자동으로 여러 클립 분할 후 ffmpeg concat.
  ex) 22초 → [15초 클립, 7초 클립] → 연결
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime

import requests

from .base import BaseVideoGeneratorAdapter, VideoGenerationResult
from ..utils.masking import mask_key, sanitize_exception_msg

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_ID = "fal-ai/kling-video/v3/standard/text-to-video"
_IMAGE_TO_VIDEO_MODEL_ID = "fal-ai/kling-video/v3/standard/image-to-video"
_MAX_CLIP_SECONDS = 15
_PROMPT_MAX_LEN = 2500
_POLL_INTERVAL_SECONDS = 30  # 폴링 간격
_POLL_TIMEOUT_SECONDS = 600  # 최대 대기 10분 (초과 시 실패 처리)
_BANNED_KEYWORDS = frozenset({"nsfw", "nude", "explicit", "porn", "graphic violence"})


class KlingAdapter(BaseVideoGeneratorAdapter):
    """
    [팀 직접 구현] fal.ai Kling V3 Standard 어댑터.
    FAL_API_KEY + fal-client 패키지 필요.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = _DEFAULT_MODEL_ID,
        upload_folder: str = "uploads",
        extend_target_seconds: int = 8,
        aspect_ratio: str = "16:9",
        cfg_scale: float = 0.5,
    ):
        self.api_key = api_key
        self.model_id = model_id
        self.upload_folder = upload_folder
        self.extend_target_seconds = extend_target_seconds
        self.aspect_ratio = aspect_ratio
        self.cfg_scale = cfg_scale
        os.makedirs(upload_folder, exist_ok=True)

    def adapter_name(self) -> str:
        return "kling"

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

        os.environ["FAL_KEY"] = self.api_key

        target = self.extend_target_seconds
        if target <= _MAX_CLIP_SECONDS:
            return self._generate_single(fal_client, prompt, target, image_url)
        return self._generate_multi(fal_client, prompt, target, image_url)

    # ── 단일 클립 ──────────────────────────────────────────────────────────────

    def _generate_single(
        self,
        fal_client,
        prompt: str,
        duration: int,
        image_url: str | None,
    ) -> VideoGenerationResult:
        # image_url이 있으면 image-to-video 엔드포인트 사용 (프레임 체이닝)
        model_id = _IMAGE_TO_VIDEO_MODEL_ID if image_url else self.model_id

        arguments: dict = {
            "prompt": prompt[:_PROMPT_MAX_LEN],
            "duration": str(min(duration, _MAX_CLIP_SECONDS)),
            "aspect_ratio": self.aspect_ratio,
            "cfg_scale": self.cfg_scale,
            "generate_audio": False,  # 오디오는 ElevenLabs + ffmpeg으로 별도 처리
        }

        # 로컬 파일 경로이면 fal.ai 스토리지에 업로드 후 URL 획득
        if image_url:
            try:
                if os.path.exists(image_url):
                    uploaded_url = fal_client.upload_file(image_url)
                    arguments["image_url"] = uploaded_url
                    logger.info("[Kling] 프레임 체이닝 — 로컬 파일 업로드 완료: %s", uploaded_url)
                else:
                    arguments["image_url"] = image_url
                    logger.info("[Kling] 프레임 체이닝 — 외부 URL 사용: %s", image_url[:60])
            except Exception as e:
                logger.warning("[Kling] image_url 처리 실패, text-to-video로 폴백: %s", sanitize_exception_msg(e))
                model_id = self.model_id
                arguments.pop("image_url", None)

        # ── submit() 으로 job 제출 (request_id 즉시 확보 → 중복 제출 방지) ──────
        try:
            logger.info("[Kling] 클립 생성 요청 (duration=%ds, image=%s, key=%s)",
                        duration, bool(image_url), mask_key(self.api_key))
            handle = fal_client.submit(model_id, arguments=arguments)
            request_id = handle.request_id
            logger.info("[Kling] 요청 제출 완료 request_id=%s", request_id)
        except Exception as e:
            return VideoGenerationResult(
                success=False,
                error_message=f"API 제출 실패: {sanitize_exception_msg(e)}",
            )

        # ── 명시적 폴링 (타임아웃 + 진행 로그) ───────────────────────────────
        result = self._poll_until_done(fal_client, request_id)
        if result is None:
            return VideoGenerationResult(
                success=False,
                error_message=f"생성 타임아웃 ({_POLL_TIMEOUT_SECONDS}초 초과) — request_id={request_id}",
            )

        try:
            video_url = result["video"]["url"]
        except (KeyError, TypeError) as e:
            return VideoGenerationResult(success=False, error_message=f"응답 파싱 실패: {e}")

        try:
            file_path = self._download(video_url, f"kling_{request_id}")
        except Exception as e:
            return VideoGenerationResult(
                success=False,
                error_message=f"다운로드 실패: {sanitize_exception_msg(e)}",
            )

        logger.info("[Kling] 클립 완료 → %s", file_path)
        return VideoGenerationResult(
            success=True,
            file_path=file_path,
            metadata={
                "model": self.model_id,
                "adapter": "kling",
                "request_id": request_id,
                "duration_seconds": duration,
            },
        )

    # ── 다중 클립 (15초 초과 시) ────────────────────────────────────────────────

    def _generate_multi(
        self,
        fal_client,
        prompt: str,
        target_seconds: int,
        image_url: str | None,
    ) -> VideoGenerationResult:
        durations = self._split_durations(target_seconds)
        logger.info("[Kling] 다중 클립 생성 시작 (%d클립, 총 %d초)", len(durations), sum(durations))

        tmp_paths: list[str] = []
        try:
            for i, dur in enumerate(durations):
                r = self._generate_single(fal_client, prompt, dur, image_url if i == 0 else None)
                if not r.success:
                    return VideoGenerationResult(
                        success=False,
                        error_message=f"클립 {i + 1}/{len(durations)} 생성 실패: {r.error_message}",
                    )
                tmp_paths.append(r.file_path)
                logger.info("[Kling] 클립 %d/%d 완료 (%ds)", i + 1, len(durations), dur)

            return self._concat_clips(tmp_paths, target_seconds)

        except Exception as e:
            return VideoGenerationResult(
                success=False,
                error_message=f"다중 클립 생성 실패: {sanitize_exception_msg(e)}",
            )
        finally:
            for p in tmp_paths:
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass

    @staticmethod
    def _split_durations(target: int) -> list[int]:
        """target초를 _MAX_CLIP_SECONDS 이하 청크로 분할."""
        durations: list[int] = []
        remaining = target
        while remaining > 0:
            durations.append(min(remaining, _MAX_CLIP_SECONDS))
            remaining -= _MAX_CLIP_SECONDS
        return durations

    def _concat_clips(self, paths: list[str], total_seconds: int) -> VideoGenerationResult:
        if not shutil.which("ffmpeg"):
            logger.warning("[Kling] ffmpeg 없음 — 첫 번째 클립만 반환 (graceful degradation)")
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(self.upload_folder, f"kling_{ts}.mp4")
            shutil.copy(paths[0], final_path)
            return VideoGenerationResult(
                success=True,
                file_path=final_path,
                metadata={"adapter": "kling", "clips": 1, "warning": "ffmpeg 없음, 첫 클립만 사용"},
            )

        concat_file = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as f:
                for p in paths:
                    # 상대 경로 → 절대 경로 변환 (ffmpeg은 concat txt 위치 기준으로 상대 경로 해석)
                    # 백슬래시 → 포워드 슬래시 변환 (ffmpeg concat 형식에서 백슬래시는 이스케이프 문자)
                    abs_p = os.path.abspath(p).replace(chr(92), "/")
                    f.write(f"file '{abs_p}'\n")
                concat_file = f.name

            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            final_path = os.path.join(self.upload_folder, f"kling_{ts}_{total_seconds}s.mp4")

            subprocess.run(
                ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
                 "-i", concat_file, "-c", "copy", final_path],
                check=True,
                capture_output=True,
                timeout=120,
            )
            logger.info("[Kling] 클립 연결 완료 → %s (%d클립, %d초)", final_path, len(paths), total_seconds)
            return VideoGenerationResult(
                success=True,
                file_path=final_path,
                metadata={"adapter": "kling", "clips": len(paths), "duration_seconds": total_seconds},
            )

        except subprocess.CalledProcessError as e:
            return VideoGenerationResult(
                success=False,
                error_message=f"ffmpeg concat 실패: {e.stderr.decode(errors='replace')}",
            )
        except subprocess.TimeoutExpired:
            return VideoGenerationResult(success=False, error_message="ffmpeg concat 타임아웃")
        finally:
            if concat_file and os.path.exists(concat_file):
                os.unlink(concat_file)

    # ── 폴링 ──────────────────────────────────────────────────────────────────

    def _poll_until_done(self, fal_client, request_id: str) -> dict | None:
        """
        request_id로 완료될 때까지 폴링.
        - 간격: _POLL_INTERVAL_SECONDS (30초)
        - 최대: _POLL_TIMEOUT_SECONDS (600초 = 10분)
        - 타임아웃 시 None 반환 (새 job 제출 없음 — 중복 과금 방지)
        """
        start = time.time()
        elapsed_log_interval = 30  # 30초마다 진행 상황 로그
        last_log = start

        while True:
            elapsed = time.time() - start

            if elapsed >= _POLL_TIMEOUT_SECONDS:
                logger.error(
                    "[Kling] 폴링 타임아웃 (%ds 초과) request_id=%s — 중복 제출 없이 실패 처리",
                    _POLL_TIMEOUT_SECONDS, request_id,
                )
                return None

            try:
                status = fal_client.status(self.model_id, request_id, with_logs=True)
            except Exception as e:
                logger.warning("[Kling] 상태 조회 실패 (재시도): %s", sanitize_exception_msg(e))
                time.sleep(_POLL_INTERVAL_SECONDS)
                continue

            # 완료
            if hasattr(status, "response") or str(type(status).__name__) == "Completed":
                try:
                    result = fal_client.result(self.model_id, request_id)
                    logger.info("[Kling] 생성 완료 (%.0fs) request_id=%s", elapsed, request_id)
                    return result
                except Exception as e:
                    logger.error("[Kling] result 조회 실패: %s", sanitize_exception_msg(e))
                    return None

            # 진행 중 — 로그 출력 (30초 간격)
            if time.time() - last_log >= elapsed_log_interval:
                logs = getattr(status, "logs", None) or []
                log_msg = logs[-1]["message"] if logs else "대기 중"
                logger.info("[Kling] 생성 중... %.0fs 경과 — %s", elapsed, log_msg)
                last_log = time.time()

            time.sleep(_POLL_INTERVAL_SECONDS)

    # ── 유틸 ──────────────────────────────────────────────────────────────────

    def _validate_prompt(self, prompt: str) -> tuple[bool, str]:
        if not prompt.strip():
            return False, "프롬프트가 비어 있습니다"
        lower = prompt.lower()
        for kw in _BANNED_KEYWORDS:
            if kw in lower:
                return False, f"금지 키워드 포함: '{kw}'"
        return True, ""

    def _download(self, url: str, name: str) -> str:
        resp = requests.get(url, timeout=120, stream=True)
        resp.raise_for_status()
        file_path = os.path.join(self.upload_folder, f"{name}.mp4")
        with open(file_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        return file_path

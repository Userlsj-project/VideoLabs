"""
Step 3: AI 영상 생성 어댑터 추상 인터페이스
특정 엔진에 종속되지 않도록 어댑터 패턴으로 설계.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VideoGenerationResult:
    success: bool
    file_path: str | None = None
    storyboard: str | None = None
    error_message: str | None = None
    metadata: dict | None = None


class BaseVideoGeneratorAdapter(ABC):
    """모든 영상 생성 어댑터가 구현해야 할 인터페이스."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        genre: str = "",
        image_url: str | None = None,
    ) -> VideoGenerationResult:
        """
        프롬프트를 받아 영상(또는 영상 생성 결과물)을 반환한다.

        Args:
            prompt: AI 영상 생성 도구에 전달할 조합된 프롬프트
            genre: 장르 힌트 (optional)
            image_url: 첫 프레임 이미지 URL (image-to-video 모드, optional)

        Returns:
            VideoGenerationResult
        """

    @abstractmethod
    def adapter_name(self) -> str:
        """어댑터 이름 반환."""

    def is_available(self) -> bool:
        """어댑터가 사용 가능한 상태인지 확인."""
        return True

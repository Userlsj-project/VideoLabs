"""
[팀 직접 구현] 음악 생성 Mock 어댑터.

API 키 없이 항상 success=True를 반환한다.
file_path=None이므로 립싱크 어댑터와 연동 시 오디오가 없는 경우로 처리된다.
데모에서 음악 생성 단계를 시각적으로 보여주되 실제 파일 불필요할 때 사용.
"""

from .base import BaseMusicAdapter, MusicGenerationResult


class MockMusicAdapter(BaseMusicAdapter):
    def adapter_name(self) -> str:
        return "mock_music"

    def generate(
        self,
        prompt: str,
        duration: float = 15.0,
        genre: str = "",
    ) -> MusicGenerationResult:
        return MusicGenerationResult(
            success=True,
            file_path=None,
            duration_seconds=duration,
            style_tags=["mock", genre] if genre else ["mock"],
            lyrics="(mock: 실제 음악 생성 없음)",
            error_message="",
            estimated_cost_usd=0.0,
            adapter=self.adapter_name(),
        )

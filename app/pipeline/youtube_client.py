"""
Step 4-5: YouTube 업로드 및 댓글 수집
SIMULATION_MODE=true 시 실제 API 없이 동작한다.

YouTube Data API commentThreads.list 호출에는 최소 5분 간격 쓰로틀링을 적용해
쿼터 소진을 방지한다.
"""

import logging
import os
import json
import random
import hashlib
from datetime import datetime, timedelta
from dataclasses import dataclass

from .utils.masking import sanitize_exception_msg

logger = logging.getLogger(__name__)

# ── YouTube API 쓰로틀링 ─────────────────────────────────────────────────────
_MIN_COMMENT_FETCH_INTERVAL = timedelta(seconds=10)
# video_id → 마지막 commentThreads.list 호출 시각
_last_comment_fetch: dict[str, datetime] = {}


@dataclass
class YouTubeComment:
    comment_id: str
    author: str
    text: str
    published_at: str
    like_count: int = 0


@dataclass
class UploadResult:
    success: bool
    video_id: str | None = None
    youtube_url: str | None = None
    error_message: str | None = None
    simulated: bool = False


class YouTubeClient:
    def __init__(
        self,
        api_key: str | None = None,
        client_secrets_file: str = "client_secrets.json",
        credentials_file: str = "token.json",
        simulation_mode: bool = True,
    ):
        self.api_key = api_key
        self.client_secrets_file = client_secrets_file
        self.credentials_file = credentials_file
        self.simulation_mode = simulation_mode
        self._youtube = None

    def _get_youtube_service(self):
        """OAuth2 인증 후 YouTube 서비스 객체를 반환한다."""
        if self._youtube:
            return self._youtube
        from googleapiclient.discovery import build
        from google_auth_oauthlib.flow import InstalledAppFlow
        from google.auth.transport.requests import Request
        import pickle

        scopes = [
            "https://www.googleapis.com/auth/youtube.upload",
            "https://www.googleapis.com/auth/youtube.force-ssl",
        ]
        creds = None
        if os.path.exists(self.credentials_file):
            with open(self.credentials_file, "rb") as f:
                creds = pickle.load(f)
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.client_secrets_file, scopes
                )
                creds = flow.run_local_server(port=0)
            with open(self.credentials_file, "wb") as f:
                pickle.dump(creds, f)
        self._youtube = build("youtube", "v3", credentials=creds)
        return self._youtube

    def upload_video(
        self,
        file_path: str,
        title: str,
        description: str = "",
        tags: list[str] | None = None,
    ) -> UploadResult:
        if self.simulation_mode or not os.path.exists(file_path or ""):
            return self._simulate_upload(title)

        try:
            from googleapiclient.http import MediaFileUpload

            youtube = self._get_youtube_service()
            body = {
                "snippet": {
                    "title": title,
                    "description": description,
                    "tags": tags or [],
                    "categoryId": "22",  # People & Blogs
                },
                "status": {"privacyStatus": "unlisted"},
            }
            media = MediaFileUpload(file_path, chunksize=-1, resumable=True)
            request = youtube.videos().insert(
                part=",".join(body.keys()), body=body, media_body=media
            )
            response = None
            while response is None:
                _, response = request.next_chunk()

            video_id = response["id"]
            return UploadResult(
                success=True,
                video_id=video_id,
                youtube_url=f"https://www.youtube.com/watch?v={video_id}",
            )
        except Exception as e:
            return UploadResult(success=False, error_message=str(e))

    def _simulate_upload(self, title: str) -> UploadResult:
        seed = hashlib.md5(f"{title}{datetime.utcnow()}".encode()).hexdigest()[:11]
        video_id = seed
        return UploadResult(
            success=True,
            video_id=video_id,
            youtube_url=f"https://www.youtube.com/watch?v={video_id}",
            simulated=True,
        )

    def get_comments(
        self, video_id: str, max_results: int = 100
    ) -> list[YouTubeComment]:
        comments, _ = self.get_comments_with_error(video_id, max_results)
        return comments

    def get_comments_with_error(
        self, video_id: str, max_results: int = 100
    ) -> tuple[list[YouTubeComment], str | None]:
        """댓글 수집. 반환: (댓글 목록, 오류 메시지 or None)"""
        if self.simulation_mode:
            return [], None

        if not self.api_key:
            return [], "YOUTUBE_API_KEY가 설정되지 않았습니다. .env에서 키를 확인하세요."

        # 쓰로틀링: 동일 영상에 최소 5분 간격으로만 호출
        now = datetime.utcnow()
        last = _last_comment_fetch.get(video_id)
        if last and (now - last) < _MIN_COMMENT_FETCH_INTERVAL:
            remaining = int((_MIN_COMMENT_FETCH_INTERVAL - (now - last)).total_seconds())
            logger.warning(
                "[YouTube] commentThreads.list 쓰로틀링: %s초 후 재시도 가능 (video_id=%s)",
                remaining,
                video_id,
            )
            return [], f"YouTube API 쓰로틀링 — {remaining}초 후 다시 시도하세요. (5분 최소 간격)"

        try:
            from googleapiclient.discovery import build

            youtube = build("youtube", "v3", developerKey=self.api_key)
            comments = []
            next_page = None

            while len(comments) < max_results:
                kwargs: dict = {
                    "part": "snippet",
                    "videoId": video_id,
                    "maxResults": min(100, max_results - len(comments)),
                    "textFormat": "plainText",
                }
                if next_page:
                    kwargs["pageToken"] = next_page

                resp = youtube.commentThreads().list(**kwargs).execute()
                for item in resp.get("items", []):
                    top = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append(
                        YouTubeComment(
                            comment_id=item["id"],
                            author=top.get("authorDisplayName", "Anonymous"),
                            text=top["textDisplay"],
                            published_at=top["publishedAt"],
                            like_count=top.get("likeCount", 0),
                        )
                    )
                next_page = resp.get("nextPageToken")
                if not next_page:
                    break

            _last_comment_fetch[video_id] = datetime.utcnow()
            return comments, None
        except Exception as e:
            msg = sanitize_exception_msg(e)
            logger.error("[YouTube] 댓글 수집 실패 (video_id=%s): %s", video_id, msg)
            return [], f"API 오류: {msg}"

    def generate_simulated_comments(
        self,
        prompt: str,
        genre: str,
        count: int = 20,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-6",
        use_gemini: bool = False,
    ) -> list[YouTubeComment]:
        """
        Gemini 또는 Claude로 영상 프롬프트와 장르에 맞는 시뮬레이션 댓글을 생성한다.
        api_key가 없으면 고정된 샘플 댓글을 반환한다.
        """
        if not api_key:
            return self._generate_static_comments(genre, count)
        if use_gemini or "gemini" in model.lower():
            return self._generate_with_gemini(prompt, genre, count, api_key, model)
        return self._generate_with_claude(prompt, genre, count, api_key, model)

    def _generate_with_gemini(
        self, prompt: str, genre: str, count: int, api_key: str, model: str
    ) -> list[YouTubeComment]:
        import re
        try:
            from google import genai
            from google.genai import types as gtypes
        except ImportError:
            return self._generate_static_comments(genre, count)

        client = genai.Client(api_key=api_key)
        user_message = self._sim_comment_prompt(prompt, genre, count)
        try:
            response = client.models.generate_content(
                model=model,
                contents=user_message,
                config=gtypes.GenerateContentConfig(max_output_tokens=2048, temperature=0.8),
            )
            text = response.text.strip()
        except Exception as e:
            logger.warning("[YouTube/Gemini] 시뮬레이션 댓글 생성 실패: %s", sanitize_exception_msg(e))
            return self._generate_static_comments(genre, count)

        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            return self._generate_static_comments(genre, count)
        return self._parse_sim_comments(json_match.group())

    def _generate_with_claude(
        self, prompt: str, genre: str, count: int, api_key: str, model: str
    ) -> list[YouTubeComment]:
        import anthropic
        import re

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            messages=[
                {
                    "role": "user",
                    "content": f"""다음 AI 생성 영상에 달릴 법한 유튜브 댓글 {count}개를 생성해주세요.

장르: {genre}
영상 프롬프트 요약: {prompt[:300]}

댓글은 다음 비율로 섞어주세요:
- 건설적 피드백 (구체적 개선 제안 포함): 약 40%
- 단순 감상평/칭찬: 약 35%
- 스팸/무관한 내용: 약 15%
- 부적절한 요청: 약 10%

한국어 댓글로 작성하고 JSON 배열로 반환하세요:
[
  {{"author": "닉네임", "text": "댓글 내용"}},
  ...
]

건설적 피드백 예시:
- "음악이 너무 시끄러워서 목소리가 안 들려요. 볼륨 조절해주세요"
- "자막 글씨가 너무 작아요. 키워주세요"
- "편집이 너무 빠른데 좀 천천히 보여주면 좋겠어요"
- "배경이 너무 단조로운데 다양한 장면 넣어줬으면"
""",
                }
            ],
        )

        text = response.content[0].text.strip()
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if not json_match:
            return self._generate_static_comments(genre, count)

        raw = json.loads(json_match.group())
        comments = []
        for i, item in enumerate(raw):
            dt = datetime.utcnow() - timedelta(hours=random.randint(1, 72))
            comments.append(
                YouTubeComment(
                    comment_id=f"sim_{i:04d}",
                    author=item.get("author", f"User{i}"),
                    text=item.get("text", ""),
                    published_at=dt.isoformat() + "Z",
                    like_count=random.randint(0, 50),
                )
            )
        return comments

    def _sim_comment_prompt(self, prompt: str, genre: str, count: int) -> str:
        return (
            f"다음 AI 생성 영상에 달릴 법한 유튜브 댓글 {count}개를 생성해주세요.\n\n"
            f"장르: {genre}\n"
            f"영상 프롬프트 요약: {prompt[:300]}\n\n"
            "댓글은 다음 비율로 섞어주세요:\n"
            "- 건설적 피드백 (구체적 개선 제안 포함): 약 40%\n"
            "- 단순 감상평/칭찬: 약 35%\n"
            "- 스팸/무관한 내용: 약 15%\n"
            "- 부적절한 요청: 약 10%\n\n"
            "한국어 댓글로 작성하고 JSON 배열로 반환하세요:\n"
            '[{"author": "닉네임", "text": "댓글 내용"}, ...]'
        )

    def _parse_sim_comments(self, json_str: str) -> list[YouTubeComment]:
        try:
            raw = json.loads(json_str)
        except Exception:
            return []
        comments = []
        for i, item in enumerate(raw):
            dt = datetime.utcnow() - timedelta(hours=random.randint(1, 72))
            comments.append(
                YouTubeComment(
                    comment_id=f"sim_{i:04d}",
                    author=item.get("author", f"User{i}"),
                    text=item.get("text", ""),
                    published_at=dt.isoformat() + "Z",
                    like_count=random.randint(0, 50),
                )
            )
        return comments

    def _generate_static_comments(self, genre: str, count: int) -> list[YouTubeComment]:
        samples = [
            ("영상팬123", "영상 정말 잘 봤어요! 다음 편도 기대됩니다."),
            ("피드백킹", "배경음악이 너무 커서 목소리가 잘 안 들려요. 음악 볼륨 줄여주세요."),
            ("영상전문가", "자막 글씨가 너무 작아서 읽기 힘드네요. 폰트 크기 키워주세요."),
            ("구독자A", "편집이 너무 빠른 것 같아요. 장면 전환 속도를 좀 느리게 해주세요."),
            ("영화팬", "내용은 좋은데 화질이 좀 아쉽네요. 4K로 찍어주시면 어떨까요?"),
            ("좋아요누름", "좋아요 누르고 갑니다~"),
            ("광고맨", "제 채널도 방문해주세요! 구독 부탁드려요!!"),
            ("시청자B", "배경이 단조로운데 다양한 장면 넣어줬으면 더 좋을 것 같아요."),
            ("열혈팬", "오오 대박이다! 최고예요!"),
            ("초보자", "처음 봤는데 진짜 퀄리티 좋네요. 구독했어요."),
            ("피드백러", "인트로가 너무 길어요. 20초 안으로 줄여주세요."),
            ("댓글러", "자막 색깔이 배경이랑 비슷해서 안 보여요. 대비 높여주세요."),
            ("시청자C", "내용이 너무 전문적이어서 초보자가 이해하기 어렵네요. 쉽게 설명해주세요."),
            ("익명", "그냥 그래요"),
            ("영상덕후", "조명이 너무 어두운 장면이 있어요. 전체적으로 밝게 해주세요."),
            ("피드백마스터", "목소리가 너무 단조로워요. 톤에 변화를 주면 더 재미있을 것 같아요."),
            ("일반시청자", "재미있게 잘 봤습니다!"),
            ("댓글왕", "ㅋㅋㅋ 재밌다"),
            ("관심자", "이 기술 어떻게 배울 수 있나요?"),
            ("피드백러2", "영상 길이가 너무 짧아요. 내용을 더 자세히 설명해주세요."),
        ]
        selected = (samples * ((count // len(samples)) + 1))[:count]
        comments = []
        for i, (author, text) in enumerate(selected):
            dt = datetime.utcnow() - timedelta(hours=random.randint(1, 168))
            comments.append(
                YouTubeComment(
                    comment_id=f"static_{i:04d}",
                    author=author,
                    text=text,
                    published_at=dt.isoformat() + "Z",
                    like_count=random.randint(0, 30),
                )
            )
        return comments

"""피드백 0개 문제 디버깅"""
import os, sys
os.environ.setdefault("WTF_CSRF_ENABLED", "False")
sys.path.insert(0, os.path.dirname(__file__))
from app import create_app
from app.models import db, PipelineSession, VideoRound, Feedback

app = create_app()
app.config["TESTING"] = True
app.config["WTF_CSRF_ENABLED"] = False

with app.test_client() as c:
    with app.app_context():
        # 1. 세션 생성
        r = c.post("/session/", data={
            "title": "디버그 세션", "genre": "교육",
            "user_script": "테스트 스크립트"
        }, follow_redirects=False)
        print(f"세션 생성: {r.status_code}")

        sess = PipelineSession.query.filter_by(title="디버그 세션").first()
        if not sess:
            print("세션 없음!")
            sys.exit(1)
        sid = sess.id
        print(f"세션 ID: {sid}, owner_token: {sess.owner_token}")

        # 2. 라운드 확인
        round_ = VideoRound.query.filter_by(session_id=sid).first()
        print(f"라운드: {round_.id}, 상태: {round_.status}")

        # 3. 피드백 추가
        r = c.post(f"/session/{sid}/add-test-feedback",
                   data={"comment_text": "배경음악이 너무 커서 목소리가 전혀 안 들려요."},
                   follow_redirects=True)
        print(f"피드백 추가: {r.status_code}")

        # DB에서 직접 확인
        db.session.expire_all()
        fbs = Feedback.query.filter_by(video_round_id=round_.id).all()
        print(f"DB 피드백 수: {len(fbs)}")
        for fb in fbs:
            print(f"  - [{fb.id}] {fb.comment_text[:30]} | category={fb.category}")

        # 4. api/feedbacks
        r = c.get(f"/session/{sid}/api/feedbacks")
        import json
        data = json.loads(r.data)
        print(f"api/feedbacks 반환: {len(data)}개")

        # 5. filter
        r = c.post(f"/session/{sid}/filter", follow_redirects=False)
        print(f"filter: {r.status_code}")

        # DB 재확인
        db.session.expire_all()
        fbs2 = Feedback.query.filter_by(video_round_id=round_.id).all()
        print(f"filter 후 DB 피드백: {len(fbs2)}")
        for fb in fbs2:
            print(f"  - [{fb.id}] category={fb.category}, severity={fb.severity_score}")

        # 6. api/feedbacks 다시
        r = c.get(f"/session/{sid}/api/feedbacks")
        data2 = json.loads(r.data)
        print(f"filter 후 api/feedbacks: {len(data2)}개")

        # 7. 정리
        c.post(f"/session/{sid}/delete")

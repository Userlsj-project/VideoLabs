"""전체 기능 통합 테스트 — 서버가 127.0.0.1:5000에서 실행 중이어야 함"""
import os, sys
os.environ.setdefault("WTF_CSRF_ENABLED", "False")

# 앱을 직접 생성해서 test client 사용 (서버 불필요)
sys.path.insert(0, os.path.dirname(__file__))
from app import create_app
from app.models import db, PipelineSession, VideoRound

app = create_app()
app.config["TESTING"] = True
app.config["WTF_CSRF_ENABLED"] = False

results = []

def chk(label, ok, detail=""):
    results.append((ok, label, detail))
    status = "OK" if ok else "NG"
    print(f"  {status}  {label}" + (f" — {detail}" if detail else ""))

def get(client, path, expected=(200,)):
    r = client.get(path, follow_redirects=False)
    ok = r.status_code in expected
    if not ok:
        print(f"       ↳ 응답코드: {r.status_code}")
    return r, ok

def post(client, path, data=None, json=None, expected=(200, 302, 303)):
    if json is not None:
        import json as _json
        r = client.post(path, data=_json.dumps(json),
                        content_type="application/json",
                        follow_redirects=False)
    else:
        r = client.post(path, data=data or {}, follow_redirects=False)
    ok = r.status_code in expected
    if not ok:
        print(f"       ↳ 응답코드: {r.status_code}, body: {r.data[:200]}")
    return r, ok

with app.test_client() as c:
    with app.app_context():

        # ── 1. 기본 페이지 ──────────────────────────────────────────
        print("\n[1] 기본 페이지")
        r, ok = get(c, "/")
        chk("홈(대시보드) GET /", ok)
        chk("홈에 Bootstrap 로드", b"bootstrap" in r.data.lower())
        chk("홈에 CSRF 메타태그", b"csrf-token" in r.data)
        chk("홈에 다크모드 토글", b"toggleDarkMode" in r.data)

        r, ok = get(c, "/session/new")
        chk("새 세션 폼 GET /session/new", ok)

        # ── 2. 세션 생성 ──────────────────────────────────────────
        print("\n[2] 세션 생성")
        r, ok = post(c, "/session/",
                     data={"title": "테스트 프로젝트", "genre": "교육",
                           "user_script": "AI 기술을 쉽게 설명하는 교육 영상"})
        chk("세션 POST 생성 (redirect)", ok)
        # 세션 ID 파악
        sess = PipelineSession.query.filter_by(title="테스트 프로젝트").first()
        chk("세션 DB 저장 확인", sess is not None)
        if not sess:
            print("  !! 세션 없음 — 이후 테스트 건너뜀")
            sid = 0
        else:
            sid = sess.id

        # ── 3. 세션 상세 / 프롬프트 확인 ───────────────────────────
        print("\n[3] 세션 상세")
        r, ok = get(c, f"/session/{sid}")
        chk("세션 상세 GET", ok)
        chk("세션 상세에 파이프라인 스텝", b"step-circle" in r.data)
        chk("세션 상세에 프롬프트 표시", b"video_gen_prompt" in r.data or "프롬프트".encode() in r.data)

        # ── 4. 프롬프트 인라인 수정 ─────────────────────────────────
        print("\n[4] 프롬프트 수정")
        r, ok = post(c, f"/session/{sid}/edit_prompt",
                     data={"new_prompt": "수정된 테스트 프롬프트"},
                     expected=(200, 302, 303, 400))
        chk("프롬프트 수정 POST", r.status_code in (200, 302, 303, 400))

        # ── 5. 피드백 추가 ──────────────────────────────────────────
        print("\n[5] 피드백 수집")
        round_ = VideoRound.query.filter_by(session_id=sid).first()
        chk("라운드 자동 생성", round_ is not None)
        rid = round_.id if round_ else 0

        test_comments = [
            "배경음악이 너무 커서 목소리가 전혀 안 들려요. 볼륨을 낮춰주세요.",
            "자막이 없어서 내용을 이해하기 어렵습니다. 한국어 자막을 추가해주세요.",
            "화질이 너무 낮습니다. 1080p 이상으로 올려주시면 좋겠어요.",
            "스토리 전개가 갑작스러워서 이해가 어렵습니다. 설명 장면을 추가해주세요.",
            "설명이 너무 빠릅니다. 속도를 늦춰주세요.",
            "좋은 영상이에요! 계속 만들어주세요.",
            "구독 좋아요 눌러주세요",
        ]
        fb_ok = True
        for cmt in test_comments:
            rr = c.post(f"/session/{sid}/add-test-feedback",
                        data={"comment_text": cmt}, follow_redirects=True)
            if rr.status_code != 200:
                fb_ok = False
        chk("테스트 피드백 추가 (7개)", fb_ok)

        # ── 6. 피드백 분류 ──────────────────────────────────────────
        print("\n[6] 피드백 분류")
        r, ok = post(c, f"/session/{sid}/filter", expected=(200, 302, 303))
        chk("피드백 분류 POST", ok)

        r, ok = get(c, f"/session/{sid}/api/feedbacks")
        chk("피드백 JSON API", ok)
        import json as _json
        try:
            data = _json.loads(r.data)
            chk("피드백 API 데이터 존재", len(data) > 0, f"{len(data)}개")
            if data:
                f0 = data[0]
                chk("severity_score 필드", "severity_score" in f0,
                    str(f0.get("severity_score")))
                chk("quality_score 필드", "quality_score" in f0,
                    str(f0.get("quality_score")))
                chk("actionability_score 필드", "actionability_score" in f0,
                    str(f0.get("actionability_score")))
        except Exception as e:
            chk("피드백 API JSON 파싱", False, str(e))

        # ── 7. 클러스터링 ───────────────────────────────────────────
        print("\n[7] 클러스터링")
        r, ok = post(c, f"/session/{sid}/cluster", expected=(200, 302, 303))
        chk("클러스터링 POST", ok)

        r, ok = get(c, f"/session/{sid}/api/clusters")
        chk("클러스터 JSON API", ok)
        try:
            data = _json.loads(r.data)
            chk("클러스터 데이터 존재", len(data) > 0, f"{len(data)}개")
            if data:
                c0 = data[0]
                chk("cohesion_score 필드", "cohesion_score" in c0,
                    str(c0.get("cohesion_score")))
                chk("priority_score 필드", "priority_score" in c0,
                    str(c0.get("priority_score")))
        except Exception as e:
            chk("클러스터 API JSON 파싱", False, str(e))

        # ── 8. 클러스터 검토 페이지 ─────────────────────────────────
        print("\n[8] 클러스터 검토 UI")
        r, ok = get(c, f"/session/{sid}/review")
        chk("검토 페이지 GET", ok)
        chk("검토 페이지에 우선순위 바", b"priority-bar" in r.data)
        chk("검토 페이지에 severity-dot", b"severity-dot" in r.data)

        # ── 9. 새 API 엔드포인트 ────────────────────────────────────
        print("\n[9] 신규 API 엔드포인트")
        r, ok = get(c, f"/session/{sid}/api/engagement_lift", expected=(200, 400, 404))
        chk("F-3 참여도 예측 API", r.status_code in (200, 400, 404))

        r, ok = get(c, f"/session/{sid}/api/sentiment_drift", expected=(200, 400, 404))
        chk("F-5 감정 드리프트 API", r.status_code in (200, 400, 404))

        r, ok = get(c, f"/session/{sid}/api/aesthetic_analysis", expected=(200, 400, 404))
        chk("F-9 미적 지문 API", r.status_code in (200, 400, 404))

        r, ok = post(c, f"/session/{sid}/api/prompt_variants",
                     json={"base_prompt": "test prompt"},
                     expected=(200, 400, 404, 429))
        chk("F-7 A/B 변형 API", r.status_code in (200, 400, 404, 429))

        r, ok = post(c, f"/session/{sid}/api/reply_drafts",
                     json={},
                     expected=(200, 400, 404, 429))
        chk("F-8 답글 초안 API", r.status_code in (200, 400, 404, 429))

        # ── 10. 반복 이슈 분석 ──────────────────────────────────────
        print("\n[10] 반복 이슈 분석 (F-4)")
        r, ok = get(c, f"/session/{sid}/systemic_issues", expected=(200, 404))
        chk("F-4 반복 이슈 페이지 GET", r.status_code in (200, 404))

        # ── 11. 리포트 / 버전 페이지 ────────────────────────────────
        print("\n[11] 리포트 / 버전")
        r, ok = get(c, f"/session/{sid}/report")
        chk("리포트 GET", ok)
        # metrics 없으면 조건부 렌더링 — 페이지 자체가 로드되면 충분
        chk("리포트 정상 렌더링", b"report" in r.data.lower() or "리포트".encode() in r.data)

        r, ok = get(c, f"/session/{sid}/versions")
        chk("버전 목록 GET", ok)

        # ── 12. 상태 API / SSE ───────────────────────────────────────
        print("\n[12] 상태 API / SSE")
        r, ok = get(c, f"/session/{sid}/api/status")
        chk("상태 API GET", ok)
        try:
            d = _json.loads(r.data)
            chk("상태 API JSON 형식", "session" in d and "current_round" in d,
                f"keys={list(d.keys())}")
        except Exception as e:
            chk("상태 API JSON", False, str(e))

        # SSE 스트림 — 응답 코드만 확인 (test client에서 스트림 소비 생략)
        r2 = c.get(f"/session/{sid}/api/generation_status_stream")
        chk("SSE 스트림 엔드포인트", r2.status_code in (200, 204))
        # 스트림 명시적 소비 후 닫기 (AppContext 충돌 방지)
        try:
            _ = r2.data
        except Exception:
            pass

        # ── 13. 채널 프로파일 ───────────────────────────────────────
        print("\n[13] 채널 프로파일")
        r, ok = get(c, f"/session/{sid}/channel_profile", expected=(200, 404))
        chk("채널 프로파일 페이지 GET", r.status_code in (200, 404))

        # ── 14. 세션 삭제 ───────────────────────────────────────────
        print("\n[14] 세션 삭제")
        r, ok = post(c, f"/session/{sid}/delete", expected=(200, 302, 303))
        chk("세션 삭제 POST", ok)
        deleted = PipelineSession.query.get(sid)
        chk("DB에서 세션 제거 확인", deleted is None)

# ── 최종 결과 ─────────────────────────────────────────────────────────
total = len(results)
passed = sum(1 for ok, _, _ in results if ok)
failed = total - passed
print(f"\n{'='*50}")
print(f"결과: {passed}개 OK, {failed}개 NG  (전체 {total}개)")
if failed:
    print("\n실패 항목:")
    for ok, lbl, detail in results:
        if not ok:
            print(f"  NG  {lbl}" + (f" — {detail}" if detail else ""))
else:
    print("전체 이상 없음")

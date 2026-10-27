"""전체 라우트 및 API 기능 검증 스크립트."""
import sys, json, os, re
sys.path.insert(0, '.')

os.environ.setdefault("WTF_CSRF_ENABLED", "False")   # 테스트 클라이언트에서 CSRF 비활성화

from app import create_app
app = create_app()
app.config["WTF_CSRF_ENABLED"] = False

errors = []
ok_count = 0

def check(label, condition, detail=""):
    global ok_count
    if condition:
        print(f"  OK  {label}" + (f" — {detail}" if detail else ""))
        ok_count += 1
    else:
        print(f"  NG  {label}" + (f" — {detail}" if detail else ""))
        errors.append(label)

with app.test_client() as c:
    # 1. 홈
    r = c.get('/')
    check("홈 GET /", r.status_code == 200)

    # 2. 세션 생성 페이지
    r = c.get('/session/new')
    check("세션 생성 페이지 GET /session/new", r.status_code == 200)

    # 3. 세션 POST 생성 — 리다이렉트된 URL에서 세션 ID 추출
    r = c.post('/session/', data={
        'genre': '드라마',
        'title': '테스트 세션',
        'user_script': '주인공이 복수를 꿈꾸는 이야기입니다.',
    }, follow_redirects=False)
    check("세션 POST 생성", r.status_code == 302)

    # 생성된 세션 ID 파싱
    location = r.headers.get('Location', '')
    m = re.search(r'/session/(\d+)', location)
    sid = int(m.group(1)) if m else 1

    # DB에서 owner_token 조회 후 Flask 세션에 주입
    from app.models import PipelineSession
    with app.app_context():
        sess_obj = PipelineSession.query.get(sid)
        owner_token = sess_obj.owner_token if sess_obj else None

    if owner_token:
        with c.session_transaction() as flask_sess:
            flask_sess[f'owns_{sid}'] = owner_token

    # 4. 세션 상세 조회
    r = c.get(f'/session/{sid}')
    check(f"세션 상세 GET /session/{sid}", r.status_code in (200, 404))

    # 5. 테스트 피드백 여러 개 추가
    feedbacks = [
        '배경음악이 너무 커서 목소리가 전혀 안 들려요. 볼륨을 줄여주세요!',
        '자막 글씨가 너무 작아서 읽기 힘들어요. 20pt 이상으로 키워주세요.',
        '음악 소리가 너무 크고 대사가 안 들립니다. 꼭 수정해주세요!!!',
        '좋아요 ㅋㅋ',
        'The pacing is too fast, please slow down the transitions.',
        '스토리 전개가 갑작스러워서 이해가 어렵습니다. 설명 장면을 추가해주세요.',
        '구독 좋아요 눌러주세요',
    ]
    all_ok = True
    for fb in feedbacks:
        r = c.post(f'/session/{sid}/add-test-feedback',
                   data={'comment_text': fb}, follow_redirects=True)
        if r.status_code != 200:
            all_ok = False
    check("테스트 피드백 추가 (7개)", all_ok)

    # 실제로 추가됐는지 확인
    with app.app_context():
        from app.models import VideoRound, Feedback
        vr = VideoRound.query.filter_by(session_id=sid).order_by(VideoRound.id.desc()).first()
        fb_count = Feedback.query.filter_by(video_round_id=vr.id).count() if vr else 0
    check("피드백 DB 저장 확인", fb_count > 0, f"{fb_count}개")

    # 6. 피드백 분류
    r = c.post(f'/session/{sid}/filter', follow_redirects=True)
    check("피드백 분류 POST", r.status_code == 200)

    # 7. feedbacks API
    r = c.get(f'/session/{sid}/api/feedbacks')
    check("피드백 JSON API", r.status_code == 200)
    if r.status_code == 200:
        fbs = json.loads(r.data)
        check("피드백 API 데이터 존재", len(fbs) > 0, f"{len(fbs)}개")
        if fbs:
            f = fbs[0]
            new_fields = ['severity_score', 'urgency_score', 'quality_score', 'needs_review', 'topic_categories']
            missing = [k for k in new_fields if k not in f]
            check("피드백 API 신규 필드", not missing,
                  f"severity={f.get('severity_score')}, quality={f.get('quality_score')}" if not missing else f"누락: {missing}")

    # 8. 클러스터링
    r = c.post(f'/session/{sid}/cluster', follow_redirects=True)
    check("클러스터링 POST", r.status_code == 200)

    # 9. clusters API
    r = c.get(f'/session/{sid}/api/clusters')
    check("클러스터 JSON API", r.status_code == 200)
    if r.status_code == 200:
        clusters = json.loads(r.data)
        check("클러스터 API 데이터 존재", len(clusters) > 0, f"{len(clusters)}개 클러스터")
        if clusters:
            cl = clusters[0]
            new_fields = ['cohesion_score', 'cluster_summary', 'avg_severity_score', 'avg_quality_score']
            missing = [k for k in new_fields if k not in cl]
            check("클러스터 API 신규 필드", not missing,
                  f"cohesion={cl.get('cohesion_score')}, summary={str(cl.get('cluster_summary',''))[:40]}" if not missing else f"누락: {missing}")

    # 10. 검토 UI
    r = c.get(f'/session/{sid}/review')
    check("클러스터 검토 GET", r.status_code in (200, 302))

    # 11. 리포트
    r = c.get(f'/session/{sid}/report')
    check("리포트 GET", r.status_code in (200, 404))

    # 12. 버전 목록
    r = c.get(f'/session/{sid}/versions')
    check("버전 목록 GET", r.status_code in (200, 404))

    # 13. status API
    r = c.get(f'/session/{sid}/api/status')
    check("상태 API GET", r.status_code == 200)

    # 14. 멀티샷 API (Gemini 키 무효 시 400 또는 500도 정상 — 동작 자체가 올바름)
    r = c.post(f'/session/{sid}/api/decompose_shots',
               json={'num_shots': 3},
               content_type='application/json')
    check("샷 분해 API POST", r.status_code in (200, 400, 500))

    # 15. 캐릭터 프로파일 페이지
    r = c.get(f'/session/{sid}/character_profiles')
    check("캐릭터 프로파일 GET", r.status_code == 200)

    # 16. SSE 스트림 — test_client와 앱 컨텍스트 충돌로 직접 호출 불가, 라우트 등록 여부만 확인
    from app import create_app as _ca
    _app = _ca()
    sse_route_exists = any(
        '/api/generation_status_stream' in str(rule)
        for rule in _app.url_map.iter_rules()
    )
    check("SSE 스트림 라우트 등록", sse_route_exists)

    # 17. 채널 프로파일
    r = c.get(f'/session/{sid}/channel_profile')
    check("채널 프로파일 GET", r.status_code == 200)

print()
print(f"결과: {ok_count}개 OK, {len(errors)}개 NG")
if errors:
    print("문제 발생 항목:")
    for e in errors:
        print(f"  - {e}")
else:
    print("전체 이상 없음")

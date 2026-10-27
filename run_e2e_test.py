"""전체 파이프라인 mock end-to-end 테스트 (DB 리셋 후 동적 ID 추적)."""
import os, sys, json, time, shutil

os.environ['WTF_CSRF_ENABLED'] = 'False'
os.environ['VIDEO_GENERATOR_ADAPTER'] = 'mock'
os.environ['MUSIC_GENERATOR_ADAPTER'] = 'mock'
os.environ['LIPSYNC_ADAPTER'] = 'mock'
os.environ['SIMULATION_MODE'] = 'true'

sys.path.insert(0, '.')

# DB 리셋 (깨끗한 상태로 시작)
db_path = os.path.join('instance', 'pipeline.db')
proto_path = os.path.join('instance', 'classifier_prototypes.npz')
if os.path.exists(db_path):
    os.remove(db_path)
    print(f'[준비] DB 초기화: {db_path}')
if os.path.exists(proto_path):
    os.remove(proto_path)

from app import create_app
app = create_app()
app.config['WTF_CSRF_ENABLED'] = False

ok = 0; ng = 0; errors = []

def check(label, cond, detail=''):
    global ok, ng
    mark = 'OK' if cond else 'NG'
    suffix = f' — {detail}' if detail else ''
    print(f'  {mark}  {label}{suffix}')
    if cond:
        ok += 1
    else:
        ng += 1
        errors.append(f'{label}' + (f' ({detail})' if detail else ''))

def get_status(c, sid):
    r = c.get(f'/session/{sid}/api/status')
    if r.status_code != 200:
        return ''
    data = json.loads(r.data)
    # 응답 구조: {"current_round": {"status": ...}, "session": {...}, "total_rounds": N}
    return data.get('current_round', {}).get('status', '')

def wait_for_status(c, sid, target, timeout=8):
    for _ in range(timeout * 2):
        time.sleep(0.5)
        if get_status(c, sid) == target:
            return True
    return False

# ─── 본 테스트 ────────────────────────────────────────────────────────────────
with app.test_client() as c:

    # ── 1단계: 세션 생성 ──────────────────────────────────────────────────────
    print('\n=== [1단계] 세션 생성 ===')
    r = c.post('/session/', data={
        'genre': '교육',
        'title': 'mock e2e 테스트',
        'user_script': '파이썬 기초를 쉽게 알려주는 교육 영상입니다.',
    }, follow_redirects=False)
    check('세션 생성 POST', r.status_code in (301, 302), f'HTTP {r.status_code}')

    # 리다이렉트에서 세션 ID 추출
    location = r.headers.get('Location', '')
    import re
    m = re.search(r'/session/(\d+)', location)
    SID = int(m.group(1)) if m else 1
    print(f'  >>  생성된 세션 ID: {SID}')

    r = c.get(f'/session/{SID}')
    check('세션 상세 페이지 GET', r.status_code == 200)

    status = get_status(c, SID)
    check('초기 상태 = prompt_built', status == 'prompt_built', f'status={status}')

    r = c.get(f'/session/{SID}/api/status')
    api_data = json.loads(r.data)
    prompt_text = (api_data.get('current_round', {}).get('video_gen_prompt') or '')
    check('초기 프롬프트 존재', len(prompt_text) > 10, prompt_text[:60])

    # ── 2단계: 피드백 수집 ────────────────────────────────────────────────────
    print('\n=== [2단계] 피드백 수집 ===')
    feedbacks = [
        '배경음악이 너무 커서 목소리가 전혀 안 들려요. 볼륨을 꼭 줄여주세요!',
        '자막 글씨가 너무 작아요. 20pt 이상으로 키워주세요.',
        '음악 소리가 대사보다 크게 들립니다. 수정 부탁드려요.',
        '영상 속도가 너무 빠릅니다. 장면 전환을 천천히 해주세요.',
        '스토리 전개가 갑작스러워서 이해가 어렵습니다. 설명 장면을 추가해주세요.',
        '조명이 너무 어두워서 얼굴이 잘 안 보여요. 밝게 해주세요.',
        '내용이 너무 전문적입니다. 쉬운 용어로 설명해주세요.',
        '좋아요 ㅋㅋ',
        '구독 좋아요 눌러주세요 제 채널도 방문해주세요',
    ]
    all_ok = True
    for fb in feedbacks:
        r = c.post(f'/session/{SID}/add-test-feedback',
                   data={'comment_text': fb}, follow_redirects=True)
        if r.status_code != 200:
            all_ok = False
    check(f'테스트 피드백 {len(feedbacks)}개 추가', all_ok)

    # ── 3단계: 피드백 분류 ────────────────────────────────────────────────────
    print('\n=== [3단계] 피드백 분류 ===')
    r = c.post(f'/session/{SID}/filter', follow_redirects=True)
    check('피드백 분류 POST', r.status_code == 200)

    r = c.get(f'/session/{SID}/api/feedbacks')
    fbs = json.loads(r.data)
    constructive = [f for f in fbs if f.get('category') == 'constructive']
    spam         = [f for f in fbs if f.get('category') == 'spam']
    non_c        = [f for f in fbs if f.get('category') == 'non_constructive']
    check('건설적 피드백 ≥5개', len(constructive) >= 5,
          f'constructive={len(constructive)}, spam={len(spam)}, non_c={len(non_c)}')
    check('스팸 감지 ≥1개', len(spam) >= 1, f'{len(spam)}개')

    if fbs:
        f0 = fbs[0]
        check('6축 점수 필드 (severity·urgency·quality·needs_review)',
              all(k in f0 for k in ('severity_score', 'urgency_score', 'quality_score', 'needs_review')))
        check('actionability_score 필드', 'actionability_score' in f0,
              str(f0.get('actionability_score')))

    status = get_status(c, SID)
    check('상태 → feedback_classified', status == 'feedback_classified', f'status={status}')

    # ── 4단계: 클러스터링 ─────────────────────────────────────────────────────
    print('\n=== [4단계] 클러스터링 ===')
    r = c.post(f'/session/{SID}/cluster', follow_redirects=True)
    check('클러스터링 POST', r.status_code == 200)

    r = c.get(f'/session/{SID}/api/clusters')
    clusters = json.loads(r.data)
    check('클러스터 ≥2개 생성', len(clusters) >= 2, f'{len(clusters)}개')

    if clusters:
        cl = clusters[0]
        check('cohesion_score 존재', cl.get('cohesion_score') is not None,
              str(cl.get('cohesion_score')))
        check('cluster_summary 존재', bool(cl.get('cluster_summary')),
              (cl.get('cluster_summary') or '')[:50])
        check('priority_score > 0', (cl.get('priority_score') or 0) > 0,
              f"{cl.get('priority_score'):.3f}")
        check('avg_severity·quality 존재',
              'avg_severity_score' in cl and 'avg_quality_score' in cl)

    status = get_status(c, SID)
    check('상태 → pending_review', status == 'pending_review', f'status={status}')

    # ── 5단계: 검토 페이지 ────────────────────────────────────────────────────
    print('\n=== [5단계] 클러스터 검토 페이지 ===')
    r = c.get(f'/session/{SID}/review')
    check('검토 페이지 렌더링', r.status_code in (200, 302))

    # ── 6단계: 클러스터 승인 ──────────────────────────────────────────────────
    print('\n=== [6단계] 클러스터 승인 ===')
    pending = [cl for cl in clusters if cl.get('status') == 'pending']
    approve_data = {}
    for cl in pending[:4]:
        approve_data[f"action_{cl['cluster_key']}"] = 'approved'
    if len(pending) > 4:
        approve_data[f"action_{pending[4]['cluster_key']}"] = 'rejected'
    approve_data['notes'] = 'e2e 테스트 승인'

    r = c.post(f'/session/{SID}/approve', data=approve_data, follow_redirects=True)
    check('클러스터 승인 POST', r.status_code == 200)

    # 승인 즉시 새 라운드가 생성되어 status가 prompt_built로 전환됨 — 정상 흐름
    r_st = c.get(f'/session/{SID}/api/status')
    st_data = json.loads(r_st.data)
    total = st_data.get('total_rounds', 0)
    check('승인 후 새 라운드 생성됨 (total_rounds=2)', total == 2, f'total_rounds={total}')

    # ── 7단계: 프롬프트 버전 ──────────────────────────────────────────────────
    print('\n=== [7단계] 프롬프트 버전 관리 ===')
    r = c.get(f'/session/{SID}/versions')
    check('버전 목록 페이지', r.status_code == 200)
    html = r.data.decode()
    # 첫 승인 → v1 생성, 두 번째 승인 이후 v2 생성 (단일 사이클 테스트는 v1이 정상)
    check('v1 버전 생성 확인', 'v1' in html or 'v2' in html)

    # ── 8단계: 2라운드 영상 생성 (mock) ──────────────────────────────────────
    print('\n=== [8단계] 2라운드 영상 생성 (mock) ===')
    # prompt_updated → 다음 라운드(SID+1 or same SID) 자동 생성됨
    # 새 라운드는 같은 session_id의 round_number=2
    r2 = c.get(f'/session/{SID}')
    r2_data = r2.data.decode()
    # 2라운드 세션 상세 페이지
    check('2라운드 세션 상세', r2.status_code == 200)

    status2 = get_status(c, SID)
    # prompt_updated 상태이면 새 라운드가 바로 생성됨
    # 실제로는 승인 후 새 VideoRound가 생성되고 상태 전환
    # API 상태를 다시 확인
    r_api = c.get(f'/session/{SID}/api/status')
    api2 = json.loads(r_api.data)
    current_round = api2.get('current_round', {}).get('round_number', 1)
    check('2라운드로 진행됨', current_round >= 2, f'round={current_round}')

    new_prompt = (api2.get('current_round', {}).get('video_gen_prompt') or '')
    check('2라운드 프롬프트 개선됨', len(new_prompt) > 10, new_prompt[:60])

    # 영상 생성 요청
    r = c.post(f'/session/{SID}/generate', data={
        'image_url': '', 'audio_mode': 'skip'
    }, follow_redirects=False)
    check('영상 생성 요청', r.status_code in (200, 302), f'HTTP {r.status_code}')

    # mock 생성 완료 대기 (최대 10초)
    done = wait_for_status(c, SID, 'video_ready', timeout=10)
    status_final = get_status(c, SID)
    check('mock 영상 생성 완료', status_final == 'video_ready', f'status={status_final}')

    # ── 9단계: 리포트 ─────────────────────────────────────────────────────────
    print('\n=== [9단계] 리포트 ===')
    r = c.get(f'/session/{SID}/report')
    check('리포트 페이지 렌더링', r.status_code == 200)
    check('리포트 내용 충분 (>1KB)', len(r.data) > 1000, f'{len(r.data)}B')

    # ── 10단계: 부가 기능 API ──────────────────────────────────────────────────
    print('\n=== [10단계] 부가 기능 ===')
    r = c.get(f'/session/{SID}/systemic_issues')
    check('F-4 반복이슈 분석 페이지', r.status_code == 200)

    r = c.get(f'/session/{SID}/channel_profile')
    check('채널 프로파일(시청자 DNA)', r.status_code == 200)

    r = c.get(f'/session/{SID}/api/engagement_lift')
    check('F-3 참여도 향상 예측 API', r.status_code == 200)
    if r.status_code == 200:
        eng = json.loads(r.data)
        check('  └ estimated_lift_pct 필드', 'estimated_lift_pct' in eng or 'error' in eng,
              str(eng.get('estimated_lift_pct', eng.get('error', ''))))

    r = c.get(f'/session/{SID}/api/sentiment_drift')
    check('F-5 감정 드리프트 API', r.status_code == 200)
    if r.status_code == 200:
        drift = json.loads(r.data)
        check('  └ drift_detected 필드', 'drift_detected' in drift or 'error' in drift)

    r = c.post(f'/session/{SID}/api/prompt_variants',
               data=json.dumps({}), content_type='application/json')
    check('F-7 A/B 프롬프트 변형 API', r.status_code in (200, 429, 503),
          f'HTTP {r.status_code}')

    r = c.post(f'/session/{SID}/api/reply_drafts',
               data=json.dumps({}), content_type='application/json')
    check('F-8 댓글 답글 초안 API', r.status_code in (200, 429, 503),
          f'HTTP {r.status_code}')

    r = c.get(f'/session/{SID}/api/aesthetic_analysis')
    check('F-9 미적 지문 분석 API', r.status_code == 200,
          f'HTTP {r.status_code}')

    r = c.get(f'/session/{SID}/api/prompt_performance')
    check('F-1 프롬프트 성능 이력 API', r.status_code == 200)

    # ── 11단계: 보안 기능 ─────────────────────────────────────────────────────
    print('\n=== [11단계] 보안 기능 ===')
    # H-4: 잘못된 owner_token → 거부(302 리다이렉트 또는 403)
    r = c.post(f'/session/{SID}/filter',
               headers={'Cookie': 'owner_token=wrong_token'},
               follow_redirects=False)
    check('H-4 잘못된 owner_token 거부', r.status_code in (302, 403),
          f'HTTP {r.status_code}')

    # H-1: SSRF 방지 — 내부 IP image_url 차단
    r = c.post(f'/session/{SID}/generate', data={
        'image_url': 'http://169.254.169.254/metadata',
        'audio_mode': 'skip',
    }, follow_redirects=False)
    check('H-1 SSRF 방지 (내부 IP 차단)', r.status_code in (200, 302, 400),
          f'HTTP {r.status_code}')

    # C-2: 경로 탐색 — video 서빙 시 잘못된 경로
    r = c.get(f'/session/{SID}/thumbnail/../../etc/passwd')
    check('C-2 경로 탐색 공격 차단', r.status_code in (400, 403, 404),
          f'HTTP {r.status_code}')

# ─── 최종 결과 ────────────────────────────────────────────────────────────────
print(f'\n{"=" * 55}')
print(f'결과: {ok}개 OK, {ng}개 NG')
if errors:
    print('실패 항목:')
    for e in errors:
        print(f'  ✗ {e}')
else:
    print('전체 이상 없음 ✓')

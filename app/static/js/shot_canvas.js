/**
 * [팀 직접 구현] 멀티샷 React Flow 노드 캔버스
 * React Flow 12 CDN UMD 방식 — Flask 구조 변경 없이 <div id="shot-flow-canvas">에 마운트.
 * fix3.md F-D §2단계 구현.
 *
 * 의존성 (detail.html에서 CDN 로드 후 이 파일을 로드할 것):
 *   react@18 UMD, react-dom@18 UMD, @xyflow/react@12 UMD
 *
 * 사용 방법:
 *   window.__SHOT_DATA__ 에 shots 배열을 설정한 뒤 이 파일을 로드하면 자동 마운트.
 *   외부 shots 변경 시 window.shotCanvasUpdate(shots) 호출.
 */
(function () {
  'use strict';

  const container = document.getElementById('shot-flow-canvas');
  if (!container) return;

  // CDN 로드 확인
  if (!window.React || !window.ReactDOM || !window.ReactFlow) {
    container.innerHTML =
      '<div class="text-muted small p-3">React Flow CDN 로드 실패 — 노드 캔버스를 사용할 수 없습니다.</div>';
    return;
  }

  const { ReactFlow, Controls, Background, useNodesState, useEdgesState } = window.ReactFlow;
  const { useState, useCallback, useEffect } = window.React;
  const { createRoot } = window.ReactDOM;

  const STATUS_COLORS = {
    pending:    { bg: '#f8fafc', border: '#94a3b8' },
    generating: { bg: '#fefce8', border: '#d97706' },
    done:       { bg: '#f0fdf4', border: '#16a34a' },
    failed:     { bg: '#fef2f2', border: '#dc2626' },
  };

  function buildNodes(shots) {
    return shots.map((s, i) => ({
      id: `shot-${i}`,
      position: { x: i * 270, y: 80 },
      data: { label: s, idx: i },
      style: {
        background: STATUS_COLORS[s.status]?.bg || '#f8fafc',
        border: `2px solid ${STATUS_COLORS[s.status]?.border || '#94a3b8'}`,
        borderRadius: 10,
        width: 220,
        padding: 0,
      },
    }));
  }

  function buildEdges(shots) {
    return shots.slice(1).map((_, i) => ({
      id: `e${i}-${i + 1}`,
      source: `shot-${i}`,
      target: `shot-${i + 1}`,
      label: '마지막 프레임',
      animated: shots[i]?.status === 'done',
      style: { stroke: '#3b82f6' },
      labelStyle: { fontSize: 10, fill: '#6ea8fe' },
    }));
  }

  // 커스텀 노드 콘텐츠
  function ShotNodeContent({ data }) {
    const { label: shot, idx } = data;
    const statusLabel = { pending: '대기', generating: '생성중', done: '완료', failed: '실패' };
    const badgeClass = {
      pending: 'bg-secondary', generating: 'bg-warning text-dark',
      done: 'bg-success', failed: 'bg-danger',
    };
    return React.createElement('div', { style: { padding: '8px 10px' } },
      // 헤더
      React.createElement('div', {
        style: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }
      },
        React.createElement('strong', { style: { fontSize: 13 } }, `샷 ${idx + 1}`),
        React.createElement('span', {
          className: `badge ${badgeClass[shot.status] || 'bg-secondary'}`,
          style: { fontSize: 10 }
        }, statusLabel[shot.status] || '대기'),
      ),
      // 마지막 프레임 미리보기
      shot.lastFrameUrl
        ? React.createElement('img', {
            src: shot.lastFrameUrl,
            style: { width: '100%', height: 60, objectFit: 'cover', borderRadius: 4, marginBottom: 4 },
            title: '이전 클립 마지막 프레임',
          })
        : null,
      // 프롬프트 미리보기
      React.createElement('div', {
        style: { fontSize: 11, color: '#64748b', overflow: 'hidden',
                 textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 196 },
        title: shot.prompt || '(프롬프트 없음)',
      }, shot.prompt || '(프롬프트 없음)'),
      // 카메라 힌트
      shot.camera
        ? React.createElement('div', { style: { fontSize: 10, color: '#94a3b8', marginTop: 2 } }, shot.camera)
        : null,
    );
  }

  const nodeTypes = {
    shotNode: function ({ data }) { return React.createElement(ShotNodeContent, { data }); },
  };

  function ShotFlowCanvas({ initialShots }) {
    const [shots, setShots] = useState(initialShots);
    const rawNodes = buildNodes(shots).map(n => ({ ...n, type: 'shotNode' }));
    const rawEdges = buildEdges(shots);

    const [nodes, setNodes, onNodesChange] = useNodesState(rawNodes);
    const [edges, setEdges, onEdgesChange] = useEdgesState(rawEdges);

    // 외부 shots 변경 반영
    useEffect(() => {
      window.shotCanvasUpdate = function (newShots) {
        setShots(newShots);
        setNodes(buildNodes(newShots).map(n => ({ ...n, type: 'shotNode' })));
        setEdges(buildEdges(newShots));
      };
    }, []);

    return React.createElement(
      ReactFlow,
      { nodes, edges, onNodesChange, onEdgesChange,
        nodeTypes, fitView: true, minZoom: 0.4, maxZoom: 2 },
      React.createElement(Controls),
      React.createElement(Background, { variant: 'dots', gap: 16, size: 1 }),
    );
  }

  const root = createRoot(container);
  root.render(
    React.createElement(ShotFlowCanvas, {
      initialShots: window.__SHOT_DATA__ || [],
    })
  );

  // 전역 업데이트 핸들러 (JS shot 배열 변경 시 호출)
  window.shotCanvasUpdate = function (newShots) {
    // 컴포넌트 내 useEffect에서 덮어씀 — 첫 렌더 전 fallback
    root.render(React.createElement(ShotFlowCanvas, { initialShots: newShots }));
  };
})();

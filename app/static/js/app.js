// 파이프라인 버튼 클릭 시 로딩 표시
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('form button[type="submit"]').forEach(btn => {
    btn.closest('form').addEventListener('submit', () => {
      btn.disabled = true;
      const orig = btn.innerHTML;
      btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>처리 중...';
      // 10초 후 자동 복구 (서버 오류 시)
      setTimeout(() => {
        btn.disabled = false;
        btn.innerHTML = orig;
      }, 10000);
    });
  });

  // 알림 자동 닫기 (4초)
  document.querySelectorAll('.alert-dismissible').forEach(el => {
    setTimeout(() => {
      const bsAlert = bootstrap.Alert.getOrCreateInstance(el);
      if (bsAlert) bsAlert.close();
    }, 4000);
  });
});

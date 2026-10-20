# 전체 사각영역 수집 상태 (2026-10-20)

현재 작업은 ACT 성능 테스트가 아니라 **RoboTwin Expert 데이터 수집**이다.
사용자 수정에 따라 이전62-success-cell/팔별5개 선택을 폐기하고 별도 cohort를 만들었다.

- Train: X15.0~25.9cm × Y±15cm의80개 셀 전부. 내부 과거0/3 셀18개도 포함.
- Val: 네 성공 모서리 사이2D 중앙17개. Test: 별도 중앙17개. 좌표 중복 없음.
- 기본 지점별 성공 목표1, 시도 예산1. 실패 셀은 로그만 남기고 데이터는 제외.
- Val/Test는 전체 Expert preflight 통과 후 독립 capture seed를 사용한다.
- Home/close0.0/Lift≥30mm/measured Stow/physics2s Hold/손목·모델 제한 유지.
- scene/table/head optical calibration/yaw 고정. 색/광원 DR만 사용.

## 확인 완료

관련 테스트 **66 passed**. 전체 계획 순회, 실패 파일 제거, 성공/실패 CSV/JSONL,
인터럽트 복구, 성공 파일 중복 수집 방지, target 증가, profile/runtime mismatch 거부 검증.

실제 GPU 부분 검증 (raytrace):
- grid00: 성공하여 Train HDF5 저장.
- grid04: pregrasp 실패, 로그 보존, HDF5 없음.
- mid00_08: preflight 및 capture 성공, Val HDF5 저장.
- mid03_08: preflight 및 capture 성공, Test HDF5 저장.
이4지점 검증은 전체 데이터 수집 완료를 의미하지 않는다.

그 후 최종 수집 설정(raster)으로 grid00 실제 저장 검증을 통과했다.
65frames, q/action 각12차원, head460×640, wrist240×320 RGB,
base-frame PCD, physics 기록과10Hz q/action의 정확한 정렬을 확인했다.
렌더러 차이는 metadata/signature에 기록된다. 기존 task의 renderer를 수정하지 않았다.

## 전체 수집 실행 중

최종 cohort는 아래 폴더다. grid00 성공 파일을 건너뛰고 `--resume`으로
남은 Train/Val/Test 전부를 순회하는 독립 프로세스를 실제로 시작했다.
대화 응답 후에도 계속 실행되며 종료 시 로그/coverage를 자동 갱신한다.
**아직 전체114지점 수집 완료를 확인한 상태가 아니다.**

```
/home/user/Documents/Codex/2026-10-20/referenced-chatgpt-conversation-this-is-an-2/work/RoboTwin/baseline_runs_gpu/rectangle_dataset
```

실행 프로세스 기록: `launch.json`. CLI 로그: 상위 `rectangle_dataset_full.log`.
현재 실제 숫자는 `coverage.json` / `dashboard.html`이 최신이다. 이 문서의
초기 상태를 완료 결과로 해석하지 않는다. 원본18개 데이터는 그대로 보존했다.

전체 순회 완료 `sweep_finished`와 모든 지점의 성공 목표 충족 `all_targets_met`는
다른 값이다. 실패 셀이 있어도 순회는 끝날 수 있고, 미충족 목록은 계속 남긴다.
실패를 성공으로 채우거나 자동으로 무한 재시도하지 않는다.

명령/재개/로그 해석은 `SWEEPICK_GRID_DATASET_RUNBOOK.md`를 따른다.
이번 단계에서 ACT 학습/폐루프 모델 재평가는 자동 실행하지 않는다.

# 2026-10-20 baseline verification status

원격 기준 SHA: 7cb64ef2c2dd496fc3bf54f5a35f75855c24172d.
코드/환경 및 데이터의 검증 상태를 구분한다. 아래 결과는 새 isolated checkout에서 실행했다.

## 완료

- 원격 기준 브랜치 SHA와 사용자 PC checkout SHA 일치 확인.
- table_shift 원본 300행 CSV 직접 검증: 186/300, 62 full-success / 38 full-failure.
- split manifest: Train 후보 62, Val 4, Test 4, XY 중복 없음.
- 관련 CPU 회귀 테스트 **56 passed** (ROS pytest 자동 plugin 로딩 비활성화).
- compact CPU fixture smoke: 실제 multimodal ACT forward/backward/optimizer,
  gradient 없는 Val, 정답 action 없는 prior 추론, parameter/buffer 불변 검사 통과.
- canonical 256 hidden / 4+4 layers / FPS512 / chunk20 CPU fixture smoke도 통과.
  이때의 수치는 **합성 데이터** 결과이며 실제 학습/파지 성능으로 해석하지 않는다.
- 사용자 PC의 세 custom ACT model 파일과 repo의 모델 byte identity 확인.
- 기본 sandbox에서는 CUDA 미노출, 격리 밖 PC 실행에서 RTX 4060 8GB,
  torch 2.4.1+cu121 및 CUDA 사용 가능 확인.
- **실제 GPU Expert 사전검증: 18/18 성공.** Train L5/R5, Val L2/R2,
  Test L2/R2, 각 지점에서 Lift ≥30mm, measured Stow 도달, 물리 500step
  (dt≈.004s, 2.000000095s) 동안 elevation/selected-gripper contact/Stow 유지.
- preflight 원본 요약은 `SWEEPICK_BASELINE_PREFLIGHT_20261020.json`.

## 진행 중

실제 시뮬레이션 에피소드 수집, 실제 데이터 학습 및 독립 폐루프 평가를 실행 중.
실험이 끝나면 이 절을 실제 결과로 대체한다.

## 검증의 한계

이 성공은 fixed World yaw와 fixed OS30A calibration의 지정된 작은 XY 집합에
대한 시뮬레이션 결과다. chassis/mount/opposite-arm/CuRobo collision coverage,
절대 calibration 정확도와 실물 배포 안전성을 검증한 것은 아니다.

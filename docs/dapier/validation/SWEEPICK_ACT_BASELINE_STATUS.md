# 2026-10-20 baseline verification status

원격 기준 SHA: 7cb64ef2c2dd496fc3bf54f5a35f75855c24172d.
코드/환경 및 데이터의 검증 상태를 구분한다. 아래 결과는 새 isolated checkout에서 실행했다.

## 완료

- 원격 기준 브랜치 SHA와 사용자 PC checkout SHA 일치 확인.
- table_shift 원본 300행 CSV 직접 검증: 186/300, 62 full-success / 38 full-failure.
- split manifest: Train 후보 62, Val 4, Test 4, XY 중복 없음.
- 관련 CPU 회귀 테스트 **57 passed** (ROS pytest 자동 plugin 로딩 비활성화).
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

## 실제 데이터 수집 및 학습 완료

- 수집 18/18 성공, 약 2.8GB. Train L5/R5 (662 frames), Val L2/R2
  (261 frames), Test L2/R2 (261 frames). 기존 데이터/체크포인트 미수정.
- Train 10개 전부에서 10Hz 센서 샘플과 physics-rate q/action 기록이 정확히 일치.
- 실제 simulated_expert 데이터, canonical ACT, CUDA, chunk20,
  3회차 × 최대20 batch/회차 = 총60 optimizer steps. 전체 epoch 학습은 아님.
- Train-only q/action mean/std, Train-only FPS512 후 PCD min/max 적합.
  Val은 eval/no_grad이며 모델 parameter/buffer 불변 검사를 통과.
  Test trajectory는 학습 loader/normalization에서 읽지 않음.

| 회차 | Train total | Val total | Val L1 | Val KL | Val prior(z=0) L1 |
|---|---:|---:|---:|---:|---:|
| 1 | .821791 | .611053 | .604348 | .670525 | .604338 |
| 2 | .605435 | .550749 | .549100 | .164904 | .549109 |
| 3 | .642524 | .547894 | .546983 | .091134 | .546977 |

각 total=L1+.01×KL, normalized loss다. Train은 서로 다른 20개의 shuffled
batch 평균이고 Val은 고정 261 sample 전체 평균이므로 두 평균의 미세한
대소관계나 Train 2→3회차 변화에 과도한 의미를 부여하지 않는다.
현재 값은 실제 데이터 smoke 학습 결과이며 충분한 학습이나 파지 성공의 증거가 아니다.

학습 실행:

```bash
python scripts/train_sweepick_baseline.py --data baseline_runs_gpu/episodes1 \
  --manifest docs/dapier/SWEEPICK_BASELINE_SPLITS_20261020.json \
  --out baseline_runs_gpu/train_small1 --device cuda --epochs 3 --steps 20 --chunk-size 20
```

## 독립 ACT 실행 시도 결과: 첫 명령에서 중단

최초 평가 0/4 성공 (L 0/2, R 0/2). 지점당 최대30초를 허용했으나,
모두 첫 명령이 관절 범위 검사를 통과하지 못했다. 실제 파지 동작을 시작하지
않았으므로 정상 폐루프 동작/파지 능력이 검증됐다고 말하지 않는다.

오류 로깅만 보강하고 같은 checkpoint/seed/XY/제약으로 재확인했다.
원래 결과도 `SWEEPICK_BASELINE_CLOSED_LOOP_INITIAL_20261020.json`에 보존했다.
재확인 결과는 `SWEEPICK_BASELINE_CLOSED_LOOP_DIAGNOSTIC_20261020.json`.
동일한 4지점의 진단 재실행이므로 8개의 독립 성능 표본으로 합치지 않는다.
정책/정규화/학습 데이터/명령 clipping을 변경하거나 Test에 맞춰 학습하지 않았다.

| Test XY (m) | 할당 팔 | 거부된 L shoulder_lift (rad) |
|---|---|---:|
| (.157778, .150000) | L | -1.941001 |
| (.188889, .016667) | L | -1.950400 |
| (.157778, -.116667) | R | -1.939415 |
| (.173333, -.116667) | R | -1.942170 |

실물/모델 범위 교집합 중 **URDF shoulder_lift 하한 -1.745329976rad**를
넘었다. 이번 거부 원인은 wrist_roll 비대칭 제한이 아니다. 네 경우 모두
policy tick=0, rollout physics_steps=0, trace의 applied=false다.
진단 trace에는 실제 초기 q, 예측 action12와 거부된 관절/명령/한계를 저장했다.
범위 초과 명령을 억지로 clipping/wrapping하여 실행하지 않았다.

Checkpoint SHA256:
`74bfb384756f990b52dc3179545c224f9c7f2621ee9b3c07f8a489fd221b5d93`.
소량 학습의 finite loss는 확인했지만 작업 성공이나 충분한 학습을 확인한 것은 아니다.

## 마일스톤 해석 및 다음 작업

- M1 Expert 전체 시퀀스: 실제 GPU 18/18 사전검증 통과.
- M2 episodic capture: 실제 18/18 수집 성공, Train 팔별5개와 독립 Val/Test 확보.
- M3 ACT loader/normalization/소량 학습: 실제 데이터로 통과.
- M4 정상적인 ACT 폐루프 작업: **미검증**. 현재 4/4가 첫 명령 검사에서 중단.

사용자는 이번 작업이 대량 수집 전에 파이프라인을 확인하는 테스트임을 재확인했다.
현재 10개 Train/60 optimizer step으로 ACT 성능을 판단하거나, 이 데이터를
정식 사전학습 데이터로 승격하지 않는다.

다음에는 기존 데이터를 보존한 채 Train/Val에서 초기 action의 범위 초과 원인을
점검한다. q/action rad·그리퍼 변환·denormalization·10Hz sample와 dense command의
시간 의미를 확인하고, Val을 기준으로 충분한 baseline 학습/실행을 검증한다.
현재 recorder는 10Hz 시점의 순간 drive target과 전체 physics-rate 명령을 모두
저장한다. 이것만으로 250Hz Expert 명령을 50Hz executor가 같은 궤적으로
재현한다는 뜻은 아니므로, 대량 수집 전 재생/temporal alignment 확인이 필요하다.
지금의 실패를 데이터 수 부족 하나로 단정하지 않는다.

그 후 고정 yaw/실측 calibration 조건의 baseline 데이터 수를 늘려 본 학습한다.
Test는 계속 고정하고 일상적인 튜닝/모델 선택은 Train/Val에서 한다.
회전·camera noise·물체 물리 랜덤화 후의 최종 사전학습 데이터는 별도 후속 단계다.

## 실제 실행 파일 위치 (로컬, Git 제외)

이번 checkout:
`/home/user/Documents/Codex/2026-10-20/referenced-chatgpt-conversation-this-is-an-2/work/RoboTwin`.

그 아래:
- `baseline_runs_gpu/preflight1/report.json`
- `baseline_runs_gpu/episodes1/{train,val,test}/episode_*.hdf5` (약2.8GB)
- `baseline_runs_gpu/train_small1/{metrics.json,policy_smoke.pt}`
- `baseline_runs_gpu/closed_loop1/closed_loop.json`
- `baseline_runs_gpu/closed_loop_diagnostic2/{closed_loop.json,trace_000.json,...}`

private calibration, 실제 episodes와 checkpoint는 원격 저장소/patch에 포함하지 않는다.
실행 과정에서 원본 `~/RoboTwin-tabletop`의 데이터/체크포인트를 덮지 않았다.

## 검증의 한계

이 성공은 fixed World yaw와 fixed OS30A calibration의 지정된 작은 XY 집합에
대한 시뮬레이션 결과다. chassis/mount/opposite-arm/CuRobo collision coverage,
절대 calibration 정확도와 실물 배포 안전성을 검증한 것은 아니다.

# RoboTwin 전체 사각영역 데이터 수집

2026-10-20 사용자 수정 요청을 반영한 현재 수집 파이프라인이다.
이 명령은 ACT 학습/정책 실행을 하지 않는다. RoboTwin Expert를 실행한다.
이전 팔별5개 smoke collector와 별도이며, 이전 18개 데이터/모델을 덮지 않는다.

## 고정 위치

- Train: 원래 Grid100 중 X=.150000~.2588889m 8행 × Y=±.15m 10열 = **80개 전부 시도**.
- 과거 3/3 성공62개뿐 아니라 내부 0/3 실패18개도 다시 시도한다.
- X=.274444/.290000m 두 행은 제외한다.
- 평가: 네 모서리가 모두 과거3/3 성공인 2D grid 정사각형/직사각형 중앙34개.
  외곽 Train 행/열 위에 놓지 않고 내부 중앙점만 쓴다.
- 34개를 팔별 좌표순으로 번갈아 **Val17 / Test17**로 고정 분리한다.
- Train / Val / Test에 동일 XY는 없다. 물체 footprint나 카메라 시야의 완전 비중첩을 의미하지 않는다.
- 평가점은 capture에 앞서 독립 seed로 Expert의 Stow/Hold까지 사전검증한다.
  사전검증 실패 지점은 capture하지 않고 그 이유를 로그에 남긴다.

계획: `SWEEPICK_GRID_DATASET_SPLITS_20261020.json`.
고정 head calibration / 고정 yaw / 40mm cube / 기존 measured Home / close0.0 /
Lift≥30mm / measured Stow / simulation physics 2초 Hold / 손목 및 모델 관절 제한을 유지한다.
시각적 색/광원 randomization만 적용한다.
새 전체 수집기는 속도를 위해 RoboTwin/SAPIEN raster camera shader를 기본 사용한다.
physics, K, extrinsic, depth/XYZ 계약은 유지한다. 이전 raytrace 렌더링을 원하면
`--renderer raytrace`를 쓴다. renderer는 signature 및 episode metadata에 저장되며
같은 출력 폴더에서 바꾸면 재개를 거부한다. 기존 task의 전역 기본값을 수정하지 않는다.

## 실행

먼저 원본 assets와 private profile을 준비한다. 기존 baseline runbook의 별도
worktree/asset helper를 이용한다. 아래는 현재 checkout에서 재현하는 예다.

```bash
cd /home/user/Documents/Codex/2026-10-20/referenced-chatgpt-conversation-this-is-an-2/work/RoboTwin
source "$HOME/RoboTwin/venv/bin/activate"
export SWEEPICK_TURTLEBOT_URDF="$HOME/Alpenj-DAPIER/2ARM_ROBOT/sim/turtlebot3_waffle_pi/turtlebot3_waffle_pi_mujoco.urdf"
export SWEEPICK_HEAD_PROFILE="$PWD/local_sweepick_headcal_baseline_20261020/head_profile.json"
PLAN="$PWD/docs/dapier/SWEEPICK_GRID_DATASET_SPLITS_20261020.json"
OUT="$PWD/baseline_runs_gpu/rectangle_dataset"

python scripts/collect_sweepick_grid_dataset.py --manifest "$PLAN" --plan-only

python -u scripts/collect_sweepick_grid_dataset.py \
  --manifest "$PLAN" --profile "$SWEEPICK_HEAD_PROFILE" --out "$OUT"
```

기본은 지점별 성공 목표1개, 지점별 시도 예산1개다. Train80 지점을 전부
한 번씩 실행하고, 평가34 지점은 preflight 통과 후 capture를 한 번 실행한다.
과거 실패 셀을 건너뛰거나 성공할 때까지 무한 seed 탐색하지 않는다.
실패 셀이 있어도 모든 위치의 예산을 소진/목표 달성하면 순회는 완료(exit0)다.
`all_targets_met=false`와 미충족 지점을 그대로 보고한다. 일부 실행/미완료는 exit2다.

## 로그 확인

출력 폴더에 실행 중에도 갱신되는 파일이 있다.

| 파일 | 내용 |
|---|---|
| `dashboard.html` | Train80 히트맵, Val/Test 목록, 성공·실패·미시도·실행 중 상태; 15초 자동 새로고침 |
| `attempts.csv` | 한 시도당 한 행: cell/XY/arm/split/stage/seed/success/failure_stage/Lift/Stow/Hold/error |
| `attempts.jsonl` | 시작/종료/중단 복구 이벤트를 append-only로 저장 |
| `coverage.csv` | 지점별 성공 수/목표/시도 여부/상태 |
| `coverage.json` | split별 집계, 전체 순회 완료와 성공 목표 충족을 구분 |
| `state.json` | 재개용 journal, 고정 manifest/profile/runtime hash, 설정 변경 이력 |
| `{train,val,test}/episode_*.hdf5` | **성공한 전체 Expert 궤적만** |

실패/중단 시 생성한 새 HDF5는 이 collector가 예약한 해당 파일만 제거한다.
실패의 seed/단계/오류/측정값은 남는다. 기존 다른 데이터는 지우지 않는다.
완료 HDF5만 있고 journal 종료 기록 전에 중단한 경우, 재개 시 metadata와
success를 확인해 복구하므로 중복 수집하지 않는다.
성공 기록이 있는데 파일이 없어졌다면 조용히 성공으로 집계하거나 대체하지 않고 중단한다.
동일 출력에 수집기 두 개를 실행하면 lock으로 거부한다.

## 이어서 수집 / 반복 수 늘리기

```bash
python -u scripts/collect_sweepick_grid_dataset.py \
  --manifest "$PLAN" --profile "$SWEEPICK_HEAD_PROFILE" --out "$OUT" --resume
```

이미 채운 지점은 건너뛴다. 중단된 시도는 로그에 보존하고 새 attempt id/seed로
재개한다. 완료된 실패는 기본 예산을 쓴 것으로 취급한다. 실패 셀에 추가 시도를
허용하려면 명시적으로 예산을 늘린다.

```bash
# 실패 지점의 시도 예산을 3으로 증가 (기존 성공 데이터는 그대로).
python -u scripts/collect_sweepick_grid_dataset.py \
  --manifest "$PLAN" --profile "$SWEEPICK_HEAD_PROFILE" --out "$OUT" --resume \
  --max-attempts-per-point 3

# 지점별 성공 목표를 5개로 확대. 기본 시도 예산은 target × max-attempts.
python -u scripts/collect_sweepick_grid_dataset.py \
  --manifest "$PLAN" --profile "$SWEEPICK_HEAD_PROFILE" --out "$OUT" --resume \
  --target-per-point 5 --max-attempts-per-point 3
```

목표/예산은 명시적으로 늘릴 수 있고 줄이면 거부한다. 반복을 늘려도 새로운
attempt seed를 쓰며 기존 episode를 덮지 않는다. split별 seed 대역이 분리되어 있다.
출력 folder가 이미 있는데 `--resume` 없이 실행하면 거부한다.
manifest/calibration/physical-capture 코드/URDF가 바뀌면 같은 cohort 재개를 거부한다.
변경된 실험은 새 폴더에서 수집한다.

개별 재점검은 `--point-id grid04`, `--splits val test`, `--limit-points 1`로
부분 실행할 수 있다. 부분 실행은 전체 계획을 축소하지 않으며 미수집 위치는
항상 coverage에 남는다. 처음부터 모든114지점이 metadata에 예약된다.

## 학습 연결

`train_sweepick_baseline.py --data "$OUT" --manifest "$PLAN" ...`는 이 HDF5
계약을 그대로 읽는다. 기존 팔별5~10개 상한을 제거했다. 최소한 양팔의 성공 Train과
독립 Val이 있어야 하며, 전체 계획의 순회 완료 전에는 학습을 거부한다.
성공으로 journal에 등록된 HDF5만 허용한다. Train-only q/action 및 PCD 통계,
Val no_grad, Test trajectory 미열람 조건은 유지한다.

실패 셀에서 반드시 성공 데이터가 생긴다고 주장하지 않는다. 수집 완료 후
`covered`와 `attempted`, `missing` 목록을 함께 보고해야 한다.
이번 요청에서는 수집 파이프라인을 구현·검증하며 ACT 재학습은 자동 실행하지 않는다.

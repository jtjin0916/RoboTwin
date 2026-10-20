# 사용자 PC 실행 및 재개 명령

원래 데이터/체크포인트에 쓰지 않는다. 각 명령은 새 출력 폴더를 요구한다.
아래 GPU 실행은 일반 PC 터미널에서 한다. 격리 환경에서는 CUDA가 보이지 않을 수 있다.

## 별도 작업 폴더 준비

```bash
set -euo pipefail
cd "$HOME/RoboTwin-tabletop"
git fetch https://github.com/jtjin0916/RoboTwin.git codex/sweepick-act-baseline-stow-20261020
git worktree add --detach "$HOME/RoboTwin-act-baseline" FETCH_HEAD
cd "$HOME/RoboTwin-act-baseline"
source "$HOME/RoboTwin/venv/bin/activate"
python scripts/link_sweepick_baseline_assets.py --source "$HOME/RoboTwin-tabletop/assets"
```

worktree 폴더가 이미 있으면 새 이름을 쓰거나 그 폴더의 상태를 먼저 확인한다.
asset helper는 빠진 파일/폴더만 원본에 symlink하고 기존 경로를 덮지 않는다.
자산을 편집하는 작업에서는 이 링크를 통해 원본을 바꾸지 말 것.
이번 Expert/수집/학습 경로는 자산을 읽기만 한다.

```bash
export SWEEPICK_TURTLEBOT_URDF="$HOME/Alpenj-DAPIER/2ARM_ROBOT/sim/turtlebot3_waffle_pi/turtlebot3_waffle_pi_mujoco.urdf"
export SWEEPICK_HEAD_PROFILE="$PWD/local_sweepick_headcal_baseline_20261020/head_profile.json"
if [ ! -f "$SWEEPICK_HEAD_PROFILE" ]; then
  python scripts/prepare_sweepick_headcal.py \
    --handover "$HOME/Downloads/sweepick_261020_handover.zip" \
    --out "$PWD/local_sweepick_headcal_baseline_20261020"
fi
test -f "$SWEEPICK_TURTLEBOT_URDF"
python -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"'
```

실제 URDF 위치가 다르면 본인 팀의 모델 경로로만 바꾼다. 임의 proxy로 대체하지 않는다.
private calibration 및 데이터는 Git에 넣지 않는다.

## CPU 회귀검증

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest \
  tests/test_sweepick_baseline.py \
  tests/test_sweepick_headcal.py \
  tests/test_sweepick_headcal_near_grid100.py \
  tests/test_sweepick_workspace_grid100.py \
  tests/test_sweepick_measured_contract.py -q
```

이 venv에는 ROS launch_testing의 pytest plugin API 충돌이 있어 자동 plugin
로딩을 끈다. 테스트의 합성 fixture는 실제 파지/실제 수집 학습 증거가 아니다.

## GPU 사전검증 → 수집 → 실제 ACT smoke → 독립 폐루프

```bash
PLAN="$PWD/docs/dapier/SWEEPICK_BASELINE_SPLITS_20261020.json"
RUN="$PWD/baseline_runs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RUN"

python scripts/collect_sweepick_baseline.py \
  --manifest "$PLAN" --profile "$SWEEPICK_HEAD_PROFILE" \
  --mode preflight --per-arm 5 --out "$RUN/preflight"

python scripts/collect_sweepick_baseline.py \
  --manifest "$PLAN" --profile "$SWEEPICK_HEAD_PROFILE" \
  --mode collect --per-arm 5 --preflight "$RUN/preflight/report.json" \
  --out "$RUN/episodes"

python scripts/train_sweepick_baseline.py \
  --data "$RUN/episodes" --manifest "$PLAN" --out "$RUN/train_smoke" \
  --device cuda --epochs 1 --steps 2 --chunk-size 20

python scripts/eval_sweepick_baseline.py \
  --manifest "$PLAN" --profile "$SWEEPICK_HEAD_PROFILE" \
  --preflight "$RUN/preflight/report.json" \
  --checkpoint "$RUN/train_smoke/policy_smoke.pt" \
  --out "$RUN/closed_loop" --device cuda --seconds 30
```

`set -e`인 같은 터미널에서 차례로 실행한다. preflight/collection 실패 시
다음 단계로 넘어가지 않는다. report에는 실패도 남고 성공만으로 분모를 바꾸지 않는다.
수집 완료 시 기본 Train은 팔별 5개, Val/Test는 각각 팔별 2개다.
학습기는 최소 Train 성공 5개/팔과 별도 Val 성공이 없으면 거부한다.
Test trajectory는 loader/normalization에서 열지 않는다.

preflight의 특정 지점 원인을 조사할 때만 `--point-id grid05`처럼 부분 실행한다.
그 결과로 다른 지점의 사전검증까지 통과했다고 해석하지 않는다.
seed는 부분 실행 전의 원래 전체 인덱스로 유지된다.

`train_smoke/metrics.json`은 posterior total/L1/KL, prior L1과 실제 optimizer
step 수를 기록한다. 기본 steps=2는 에피소드 길이가 아니라 epoch당 최대 batch 수다.
완전한 epoch 학습이 필요하면 새 폴더에 `--steps 100000 --epochs 10`처럼
명시적으로 늘린다. 먼저 smoke 경로가 통과하는지 확인한다.
compact CPU 모델은 `--compact-smoke --device cpu`이고 transformer 축소형임이
표시된다. 기본 GPU 경로는 256 hidden / 4+4 layers / FPS512인 원래 모델이다.
ImageNet weight 다운로드 없이 새 backbone으로 시작하며 기존 가중치를 덮지 않는다.

폐루프 결과 `closed_loop/closed_loop.json`에는 전체 및 팔별 성공/시도 수와
seed/XY별 오류가 남는다. `trace_*.json`은 q와 action의 feedback 실행 기록이다.
실패를 뺀 성공률을 만들지 말고, 4개 고정 미학습 지점의 작은 평가임을 유지한다.
학습 loss의 낮음만으로 파지 성공을 결론내리지 않는다.

## 분할 계획 재현 (검증용 새 파일에만)

```bash
python scripts/plan_sweepick_baseline.py \
  --attempts "$HOME/RoboTwin-tabletop/data_sweepick_headcal_table_shift/attempts.csv" \
  --out "$RUN/reproduced_split.json"
cmp "$PLAN" "$RUN/reproduced_split.json"
```

원본 300행, 각 셀 repeat=0/1/2, grid XY/arm/seed, 186/300 및 62/38 결과를
모두 확인한다. 원본 파일을 trajectory로 변환하는 명령이 아니다.

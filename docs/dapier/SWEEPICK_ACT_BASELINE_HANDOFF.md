# SWEEPICK / DAPIER ACT baseline 실험 및 인수인계

기준일: 2026-10-20. 이 파일과 고정 split JSON이 저장소의 지속 가능한 결정 기록이다.
채팅의 계정 메모리를 수동 갱신했다는 뜻이 아니다.

## 현재 목표와 완료 기준

이번 데이터는 최종 사전학습 데이터가 아니다. 현재 커스텀 ACT가 실제로
학습/검증/독립 폐루프 실행까지 연결되는지 작은 베이스라인으로 확인한다.

1. M1: 고정 yaw/고정 실측 head calibration에서 양팔 Expert의
   Home → Pre-grasp → Grasp → Close → Lift → Stow → Hold → success를 구현한다.
2. M2: 팔별 성공 Train Expert 5~10개를 확보하고, 미학습 XY의 Val을 따로
   수집한다. Test XY는 고정하며 학습/통계 적합에서 읽지 않는다.
3. M3: 실제 수집 데이터로 ACT Train/Val loader와 소량 학습을 통과한다.
   Train/Val loss의 L1, KL, total 및 정답 action 없는 prior(z=0) L1을 기록한다.
4. M4: Expert 동작을 호출하지 않는 별도 ACT 폐루프 실행에서 파지 후
   Stow/Hold 성공률을 분모와 팔별 결과로 보고한다. 그 전에는 Val loss를
   작업 성공률이나 실물 성능의 증거로 해석하지 않는다.

소량 학습은 학습 가능성 확인용이다. 단 한 번의 optimizer step만으로
작업 성공을 기대하거나 성공률을 추정하지 않는다. 이후 소규모 반복 학습은
별도 새 출력 폴더에 수행하고 고정 Test를 보고 매번 튜닝하지 않는다.

## 기준 저장소와 출처

- 저장소: https://github.com/jtjin0916/RoboTwin
- 기준 브랜치: `codex/sweepick-headcal-near-grid100-20261020`
- 기준 커밋: `7cb64ef2c2dd496fc3bf54f5a35f75855c24172d`
  (원격 브랜치 SHA와 로컬 사용자 PC HEAD가 일치함을 확인).
- 구현 브랜치: `codex/sweepick-act-baseline-stow-20261020`.
- 사용자 PC 원본: `~/RoboTwin-tabletop`; 가상환경: `~/RoboTwin/venv`.
- 사용자 확인 후 찾은 정확한 CSV:
  `~/RoboTwin-tabletop/data_sweepick_headcal_table_shift/attempts.csv`.
- CSV SHA256: `50cc4b29af5b4a2bdac0213616cd608165c8946f5044a16fb3029849cbf2b292`.
- 실측 진단 결과: 100 XY × 3회 = 300시도, Lift 186/300 = 62%,
  3/3 성공 셀 62개, 0/3 실패 셀 38개. 부분 성공 셀 없음.
- `data_sweepick_headcal_near_grid100/attempts.csv`는 100시도·68성공의
  다른 파일이므로 대체하지 않는다. CSV는 trajectory 데이터가 아니다.
- `/mnt/data/Pasted text(20261020-081826).txt`는 이 환경에서 찾지 못했다.
  미열람 로그를 확인했다고 주장하지 않는다.
- 사용자 PC의 `scripts/scan_sweepick_headcal_table_shift.py`는 로컬에서만
  `Task.TABLE_CENTER_WORLD_XY=(0,-.25)`를 적용한다. 새 task는 동일 중심을
  클래스에 명시하여 이 로컬 monkey patch에 의존하지 않는다.
- Stow 원문: `~/Downloads/sweepick_261020_start_carry_pose.md`를 직접 열람.
  실물 손목 제한 원문도 10/20 handover에서 확인했다.

## 고정 실험 조건

World 테이블 중심 XY=(0,-0.25)m, 2×2m. TurtleBot `base_footprint`는
X 전방, Y 좌측, Z 위. World에서 base 원점은 (0,-.2539,.5)m,
base yaw는 +90°. 표면/카메라/로봇 변환을 중복 적용하지 않는다.

블록은 40mm cube, 기존 진단과 같은 고정 World quaternion [1,0,0,0].
`yaw_rad:0`은 이 고정 World 배치의 yaw다. 회전 랜덤화 없음.
실측 OS30A depth optical extrinsic와 전체 K를 고정하고 위치/각도/높이
camera randomization을 금지한다. 실제 head는 depth/IR only이며 RGB가 아니다.
IR 영상은 이번 수집 경로에 없고, head depth와 base-frame XYZ만 사용한다.
시뮬레이션 depth가 실물 센서의 noise/holes를 재현한다는 뜻은 아니다.

블록 색, 테이블/바닥 색, 벽 배경 색과 광원을 episode마다 랜덤화한다.
기하, 질량, 크기와 yaw는 바꾸지 않는다. textured background/clutter는
필수 외부 texture 의존성을 피하기 위해 이번 task에서 사용하지 않는다.
NumPy와 별도 visual RNG를 episode seed로 고정하고 실제 색/seed를 저장한다.
카메라 extrinsic/depth noise DR는 이 단계에 포함하지 않는다.

workspace는 X≤.26m, |Y|≤.15m. 원본 grid X=.2744/.290m는 제외한다.
3/3 성공 셀 62개가 모두 이 범위에 있다. 내부 실패 셀은 재검증 전 편입 금지.
PCD crop은 X=[.13,.32], Y=[-.26,.26], Z=[.005,.40]m, 모두 base 좌표다.
X=.15m의 40mm 블록은 앞면이 X=.13m 경계와 정확히 만나 여유가 0이다.
이 Train 지점을 넓은 가시성 여유가 있는 셀로 해석하지 말 것.
Val/Test는 성공한 같은 팔의 X 방향 이웃 셀 사이 중간점을 예약하여
X 경계에 조금 더 여유를 둔다. 예약은 Expert 성공 확인과 구별한다.

## 누출 없는 분할과 사전검증

`SWEEPICK_BASELINE_SPLITS_20261020.json`에 전체 Train 후보 62개,
Val 4개(팔별 2), Test 4개(팔별 2)를 고정한다. XY 중복을 거부한다.
Train은 검증된 grid 중심, Val/Test는 grid 사이에 예약한 별도 XY다.
기본 첫 수집은 후보 범위에 걸쳐 고른 Train 팔별 5지점이다.
`--per-arm 10`이면 팔별 10지점으로 확대한다. 다른 설정으로 일부 수집한
에피소드를 섞어 10개를 초과하는 데이터셋은 이번 baseline gate에서 거부한다.

Train/Val/Test seed base는 110000/220000/330000으로 독립 고정한다.
preflight는 별도 +1000000, 독립 ACT Test는 +2000000이다.
CSV/manifest/head profile/코드 fingerprint를 함께 묶고, 전체 Expert가
성공한 지점의 preflight evidence가 없으면 본 수집을 거부한다.
preflight는 XY feasibility 확인이지 Test ACT 학습 데이터가 아니다.
좌표/에피소드의 중복을 막는 분할이며, 가까운 XY의 블록 영역이나 카메라
시야가 겹치지 않는다거나 넓은 장면 분포까지 독립이라는 뜻은 아니다.
기본 수집은 모든 split의 선택 지점에서 단 한 번씩 시도하고 실패도 기록한다.
성공을 채우려고 무제한 seed 검색이나 실패 숨김을 하지 않는다.
`--point-id`는 진단용 부분 실행이며 누락 지점은 검증된 것으로 보지 않는다.

## Expert와 기계 제약

Home은 측정 config의 기존 homestate를 사용하며 관측 q의 도달 오차를 확인한다.
Close는 기존 RoboTwin `pos=0.0` 그대로 유지한다. 이를 실물 관절 각도나
블록을 쥔 실측 개방량 1.584rad와 같다고 가정하지 않는다.

Stow 5DOF (rad):

| 팔 | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll |
|---|---:|---:|---:|---:|---:|
| L | -.0629 | -1.6122 | 1.5616 | -.0184 | -1.6210 |
| R | -.0453 | -1.6153 | 1.5585 | -.0967 | -1.5106 |

Stow는 joint-space planner 경로를 사용하고, Stow/Hold 중 그리퍼에 새 명령을
보내지 않는다. 모바일 Carry 주행은 없다. 모든 dense 경로의 모든 waypoint와
매 physics step 직전 drive command를 검사한다. wrist_roll 범위는
L [-2.744,.957], R [-2.744,.994]rad. 다른 4관절은 원문의 서보 lo/hi,
offset, sign(+), 5tick 여유로 계산한 실물 범위와 모델 범위의 교집합을 쓴다.
상한 넘는 경로를 wrap/clip하여 통과시키지 않는다. planner가 제한을 벗어난
경로를 제안하면 해당 시도를 실패로 남긴다. 새 planner 최적화는 별도 과제다.

Lift는 상승 ≥.030m, Stow는 관측 5관절 최대 오차 ≤.025rad.
Hold는 실제 scene timestep만큼 물리를 2초 이상 진행한다(`time.sleep` 없음).
각 step마다 Stow 도달, 초기 블록 중심보다 ≥.03m 높이, 선택한 팔의
그리퍼 entity와 블록 접촉을 모두 확인한다. 연속 유지 실패는 success가 아니다.
접촉을 엄격히 요구하므로 GPU에서 실제 접촉 entity와 간헐 contact 문제도
실험 증거로 점검해야 한다. 단순히 chassis 위에 얹힌 블록은 성공이 아니다.
차체/마운트/반대팔/CuRobo 충돌 프록시는 완전 검증되지 않았다.
이 코드나 파지 성공은 충돌 안전성/실물 배포 가능성을 보장하지 않는다.

## 저장 계약과 실제 ACT 호환

새 task와 저장기는 원래 RoboTwin 수집 경로를 수정하지 않는다.
HDF5는 매 시도마다 새 파일을 만들고 실패도 보존한다. 중단 파일은
`episode_*.partial.hdf5`, 완료 파일은 `episode_*.hdf5`이며 성공/완료 속성으로
학습 포함 여부를 결정한다. 실패나 중단 파일을 성공으로 대체하지 않는다.

- state12: L 5개 관절의 실제 `entity.get_qpos()` + 실제 motor gripper q,
  R 5개 관절 + 실제 motor gripper q. state14/state16을 임의로 붙이지 않는다.
- action12: 같은 순서의 실제 `joint.get_drive_target()` (모두 model rad).
- 별도 `gripper/requested_robotwin_normalized`: 기존 0..1 그리퍼 명령.
- `control/*`: physics rate(기본 250Hz)의 timestamp/q/action/요청 gripper/phase.
- 10Hz 샘플: physics 직전 같은 timestamp의 observed q, 현재 drive target,
  head depth(m)/valid mask/XYZ(base), 양쪽 wrist RGB와 block World XYZ.
- metadata: XY/grid id/arm/split/seed, 고정 head profile와 hash, visual colors,
  manifest/code hash, 실제 dt와 policy Hz. 최종 success/phase별 실패 정보 저장.

기존 `Robot.get_*_arm_jointState()`는 drive target을 반환하고,
`get_*_arm_real_jointState()`조차 gripper에는 요청값을 붙인다. 이를 observed q로
오인하면 안 된다. 새 저장기는 active joint index로 실제 그리퍼 q도 읽는다.
현재 measured config의 gripper_scale은 [2.2028,0.0]이다.
Close(0.0)와 실제 q를 따로 기록하고 폐루프는 동일 scale의 역변환으로 명령한다.
실물 각도/개방량 매핑 문제를 여기서 조용히 교정하지 않는다.

실제 커스텀 `DapierACTMultimodal`, `model_spatial`, `model.py`를 재사용한다.
사용자 PC `~/DAPIER-ACT/dapier_act`의 세 모델 파일과 byte-identical임을 확인했다.
그 저장소 loader에는 추가 RGB decode 모드가 있지만 이 RoboTwin의 loader는
historical BGR→RGB 모드다. 새 writer는 RGB→BGR→JPEG, loader는
JPEG→BGR→RGB로 짝을 맞춘다. legacy XPolicyLab encoded corpus를 섞지 않는다.

Wrist는 native 240×320 RGB uint8 → CHW float /255; resize와 ImageNet
mean/std 없음. JPEG의 미세 손실은 허용한다. 색 순서는 회귀 테스트로 확인한다.
Head는 XYZ뿐이고 head RGB token을 만들지 않는다. PCD는 이미 base이므로
World→Task 기본 변환을 다시 하지 않는다. metric FPS(512/8192 candidates)
뒤 train-only 축별 min/max를 fit하고 기존 DP3 encoder 안에서 [-1,1]로
정규화한다. Val 범위를 fit에 넣거나 clipping하여 숨기지 않는다.
q/action mean/std도 Train만 fit(std 최소 .01). Test trajectory는 열지 않는다.

Chunk20은 10Hz 데이터에서 2초 horizon. 마지막 chunk의 padding을 mask한
L1 + .01×KL을 사용한다. 현재 CVAE posterior 평가는 GT action을 encoder에
넣고 latent sampling을 하므로, 별도로 prior z=0 L1도 보고한다.
Val은 `eval()` + `no_grad()`이고 모든 parameter/buffer가 변하지 않았음을 검사한다.
학습에 사용한 것은 canonical 12/12 모델이며 real state14/16 정책의 가중치와
무조건 호환된다는 의미가 아니다. 기존 checkpoint를 덮거나 자동 로드하지 않는다.

독립 폐루프는 Test의 관측으로 chunk를 다시 예측하는 10Hz receding horizon,
첫 action을 50Hz executor에서 유지하고 물리 step을 진행한다. Expert trajectory,
정답 action, grasp/stow target을 정책 입력으로 쓰지 않는다. Stow target은 외부
평가기준에만 쓰며, 연속 2초 유지로 성공을 판정한다. 한정된 4지점 결과는
작은 baseline 성공률이며 넓은 workspace 일반화/실물 강인성 증거가 아니다.

## 다음 단계 (이번 구현에서 금지)

1. 물체 yaw 랜덤화 및 실물 wrist_roll 비대칭 제약을 포함한 파지 계획 검증.
2. 실측 오차에 근거한 camera extrinsic/depth noise 랜덤화.
3. 물체 종류·크기·질량/마찰 등 물리특성 랜덤화.

위 단계들을 구현하고 검증한 후 확장한 **최종 데이터셋**을 커스텀 ACT의
정식 사전학습 데이터로 사용한다. 현재 소량 baseline을 최종 데이터로 승격하지 않는다.

## 재개와 검증 증거

실행 명령은 `SWEEPICK_ACT_BASELINE_RUNBOOK.md`를 따른다.
실행/검증 현황은 `validation/SWEEPICK_ACT_BASELINE_STATUS.md`에 기록한다.
새 세션은 먼저 이 문서, split manifest, status를 읽고 마지막 검증 이후 변경분을
확인한다. 과거 테스트 통과를 새 코드 검증으로 대체하지 않는다.

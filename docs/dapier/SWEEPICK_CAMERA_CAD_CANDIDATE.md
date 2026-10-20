# SWEEPICK 카메라 CAD 후보 모델 인수인계 (2026-10-20)

이 결과는 **실제 STEP을 검사해 만든 조립 후보**다. 실제 결합 방향과 depth 광학 원점을 확정한 최종 보정 모델은 아니다. 기존 데이터/실행 설정을 대체하지 않는다. 광학 중심이 null이면 optical TF와 SAPIEN 카메라 extrinsic 파일을 생성하지 않는다.

후속 사용자 확인: 아래판 긴 방향=X, OS30A 후면 M3 직접 체결, 원하는 시선=수평 전방에서 하향 25도. 조립 사진과 팀원의 depth 광학 보정자료는 대기 중이다. `user_confirmed` 필드에 기록했다. 보정값을 채우더라도 실제 모델 각도가 이 요구와 1도 넘게 다르면 검토 완료 상태의 SAPIEN extrinsic 생성을 거부한다.

기준 소스: `jtjin0916/RoboTwin`, `codex/sweepick-workspace-grid100-20261019`, commit `0920801b210293f44b32570b48e6047111ee9b9a`. 2026-10-20 원격 head와 일치함을 확인했다. 새 파일만 추가한다. ROS2 원본 TurtleBot 및 팀의 measured arm base STL은 저장소에 없으므로 로컬 파일을 입력받는 방식이다.

## CAD 검사 결과

원본은 `assets/sweepick_camera_candidate/cad/`에 보존했다. SHA256, 솔리드 수, 체적, 경계는 `cad_audit.json`, 모든 면의 법선/영역/경계 및 원통축은 두 `*_faces.json`에 기록했다. 면 index는 이 버전의 importer 결과이며 원본 STEP entity 번호와 다르다.

두 파일의 활성 길이 단위는 mm이며 OpenCascade로 읽은 치수와 일치한다. 메시 좌표는 m로 변환해 저장했고 URDF scale은 모두 1이다. STEP에는 SI metre 보조 컨텍스트도 있으므로 문자열에 mm가 있다는 사실만으로 단위를 정하지 않았다. 사용자 치수, BRep 결과, mesh 경계/체적 검증을 함께 사용했다.

| 항목 | CAD 검사값 (mm) | 의미 |
|---|---:|---|
| 아래 전체 경계 | X 6.812502..86.812502; Y -19.024998..92.975002; Z 0..231.562492 | 80 × 112 × 231.562492, 유효 솔리드 1개 |
| 아래 원점 | (46.81250222347442, 36.97500175621825, 0) | 아래판 하단 사각 경계 중앙, 면 22 중심과 일치 |
| 아래판 두께 | 15 | 바닥 Z=0, 윗면 Z=15 |
| 아래판 체결 | Ø4 관통 4개, X/Y 피치 60/92 | CAD 중심 (16.812502 또는 76.812502, -9.024998 또는 82.975002) |
| 아래판 윗면 카운터보어 | Ø7.5, 깊이 1 | Z=14..15; Ø4 관통부는 Z=0..14 |
| 아래 기둥 외곽 | X 28.499996..65.125019, Y 24.274982..49.674973 | 약 36.625 × 25.400 |
| 아래 어깨면 | Z=223.712455750 | 사용자 223.712는 전체 최고점이 아니라 이 면 |
| 결합 돌기 | Z=223.712456..231.562492 | 높이 7.850037 |
| 위 전체 경계 | X -63.117993..-15.494112; Y -1.199890..176.569532; Z -30..60 | 47.623881 × 177.769422 × 90, 유효 솔리드 2개 |
| 위 직선 기둥 | Y=-1.199890..166.400116 | 길이 167.600006; 기둥 끝의 돌출/플레이트는 별도 |
| 위 기둥 외곽 | X=-63.117993..-37.717992; Z=-3.125..33.499001 | 25.400 × 36.624001 |
| 위 하단 포켓 축 구간 | Y=-1.199890..6.798111 | 깊이 7.998001, 아래 돌기보다 약 0.148mm 깊음 |
| 카메라 플레이트 | 90 × 25.5 × 5 | 위 STEP의 두 번째 솔리드, 모서리 R1 |
| 플레이트 관통 홀 | Ø3, 중심 간격 45 | 면 1188/1189, 축 (0.422618,-0.906308,0), CAD Z=-7.5/37.5 |

기둥 표면 상당 부분은 삼각형 평면으로 저장되어 있다. 원통으로 인식되는 면 전부를 홀로 해석하지 않았다. 아래판 외곽 R2, 카운터보어 R3.75 등도 원통면이다. 홈은 포켓/돌기 형상으로 존재하며, STEP 자체에 조립 mate나 끼움 공차/접착/체결 조건은 없다.

### 후보 결합과 유효 높이

아래 CAD X→로봇 -Y, CAD Y→로봇 +X, CAD Z→로봇 +Z로 둔다. 아래판 112mm 방향이 로봇 전후 X축인 것은 **사용자가 후속 확인했다**. 위 CAD X→로봇 +X, CAD Y→로봇 +Z, CAD Z→로봇 -Y로 둔다. 위 기둥 저면 중앙을 아래 기둥 어깨면에 놓았다. 위 datum은 CAD `(-50.417992592, -1.199890137, 15.187000275)`다.

이 배치에서 BRep 침범 부피는 0 mm³이며 두 부품의 접촉거리는 수치 정밀도 수준이다. 0.15mm 추가 하강 시 약 97.342mm³ 침범이 생긴다. 위 부품을 기둥축으로 180도 돌린 후보도 침범 없이 결합된다. 따라서 기하학적 비침범만으로 조립 yaw를 확정할 수 없다. `fit_probe.py`로 재현한다.

- 위 기둥 끝 높이: 223.712456 + 167.600006 = **391.312462mm** (mount datum 기준).
- 위 플레이트 포함 전체 최고점: **401.481878mm** (mount datum 기준).
- base_footprint 기준 최고점: 123 + 401.481878 = **524.481878mm**.
- 두 부품의 전체 외곽 길이를 더하면 231.562492 + 177.769422 = 409.331915mm이지만, 후보 겹침 7.850037mm를 빼면 401.481878mm가 된다.

이는 제안된 shoulder seating의 결과다. 실제 유효 높이를 확정했다는 뜻이 아니다. 위 STEP의 두 솔리드를 하나로 합치지 않고 각각 mesh로 내보냈다.

## OS30A 근거와 한계

검토 문서: 제조사 EAI 작성 **OS30A Data Sheet**, DOC 01.13.00A100, V1.0 (2022-09-01), section 3.2/4.1/7.1, PDF 물리 페이지 6/7/10. [배포사 보관 원문](https://static.generation-robots.com/media/depth-camera-ydlidar-os30a-datasheet.pdf). [제조사 다운로드 페이지](https://www.ydlidar.com/service_support.html)는 검색 색인에 OS30A 자료가 있지만 이번 접속은 HTTP 404였다. 제조사 작성 문서의 배포사 사본이며 최신 하드웨어 revision까지 확인한 것은 아니다. 일반 사용자 매뉴얼의 RGB/color stream 설명을 OS30A에 적용하지 않았다. OS30A용 공식 개발 매뉴얼/SDK depth 원점 정의는 확보하지 못했다.

문서 확인값: 바디 90 × 25.5 × 25mm, USB2.0 Type-C screw lock, depth 1280×920 / 640×460, HFOV 78±3°, VFOV 60±3°, 범위 0.2..2.5m, IR 카메라 2개와 projector 1개, baseline 45mm. RGB 센서는 구성표에 없다. Fig 5 후면 M3 깊이 3mm 홀 간격 45mm가 CAD 플레이트 Ø3 45mm 패턴과 일치한다. 다른 뷰의 34mm 홀 쌍과 end-view 홀은 후면 45mm 패턴과 혼동하면 안 된다.

홀 패턴 일치만으로 케이블 간섭, 실제 screw 길이, 바디 방향, 사용할 체결면, 광학 중심이 확정되지는 않는다. Fig 5의 렌즈 위치는 외관/기계도 정보이며 depth의 rectified virtual camera 또는 entrance pupil의 바디 기준 3D 원점을 지정하지 않는다. 좌/우 IR 중 depth 기준이 어느 쪽인지, 바디 표면에서의 깊이 offset과 SDK frame 회전은 미확정이다. body 앞면/렌즈 표면/중점을 optical origin으로 채우지 않았다. 공칭 FOV로 얻은 K는 실측 intrinsic 보정을 대신할 수 없다.

## TF와 25도 부호

모든 변환은 `T_parent_child`: child 좌표의 점을 parent로 옮기는 column-vector 변환이다. ROS body 축은 +X 전방,+Y 좌측,+Z 위. URDF RPY는 `Rz(yaw) Ry(pitch) Rx(roll)`.

| TF | 후보 값 | 상태 |
|---|---|---|
| base_footprint → mount_bottom | xyz=(-0.060,0,0.123), R=I | 위치는 사용자 지정; mesh yaw 방향은 후보 |
| mount_bottom → mount_top | xyz=(0,0,0.223712455750), R=I | shoulder seating 후보; mesh에 CAD 회전 baked |
| mount_top → camera_link | xyz=(0.023697382247,0.000187000275,0.168178426669), pitch=+65° | CAD face 1187에 바디 후면 중앙을 직접 부착한다는 가정 |
| camera_link → camera_optical_frame | xyz=null; convention rpy=(-π/2,0,-π/2) | 원점 미확정; SDK rectification 회전도 확인 필요 |

camera_link는 후보 후면 체결면의 중앙이다. 바디 proxy는 이 프레임에서 x=0..25.5mm, y=±45mm,z=±12.5mm다. CAD face 1187 normal `(sin25,-cos25,0)`을 조립 회전하면 base 방향 `(sin25,0,-cos25)`가 된다. 이는 **수평 전방에서 하향 65도, 수직 아래에서 전방으로 25도**다. 평면 경사와 법선 경사를 동일한 숫자로 쓰면 25/65도가 뒤바뀐다. 반대 면/반대 조립 yaw에서는 방향도 달라진다. 사용자는 후속 답변에서 **수평 전방(XY 평면)에서 하향 25도**임을 확인했다. 따라서 현재 CAD 후면 직접부착 후보의 65도와 불일치한다. 조립 사진으로 실제 회전/결합 방향과 플레이트 면을 확인하기 전까지 후보 65도를 실제값으로 승격하거나, 형상과 어긋나게 바디만 25도로 돌리지 않는다.

표준 optical 축은 +Z 시선,+X 영상 오른쪽,+Y 영상 아래다. 표준 변환에서 optical Z→camera X, optical X→-camera Y, optical Y→-camera Z. SAPIEN은 +X forward,+Y left,+Z up이므로 `forward=R_WO[:,2]`, `left=-R_WO[:,0]`. World의 measured 배치는 `R_WB=Rz(+90°), t_WB=(0,-0.2539,tabletop_world_z)`다. table 높이 변화는 한 번만 적용한다.

## 기존 코드 조사

- `assets/embodiments/so101-pgripper-sweepick-measured/config.yml`: arms base 좌표는 좌 (-.060,+.150,.123), 우 (-.060,-.150,.123), World yaw +90°. 원래 arm 홈/joint/gripper 설정을 보존했다.
- 같은 config의 static head World 위치는 (-.022067149296,-.286910567080,1.012937731237). base 좌표는 (-.033010567080,+.022067149296,.512937731237). forward는 base에서 약 (.349188,.004464,-.937042), 하향 약 69.6°. 기존 주석도 STILL PROVISIONAL이다. 이 값을 CAD 후보에 억지로 맞추지 않았다.
- `_camera_config.yml`: calibrated fovy=58.7°,640×460. 이 파일은 기존 실험의 근사값이며 SDK K 전체를 구현한 것은 아니다. 새 preview는 공칭 fovy=60°,near=.2,far=2.5를 사용한다. 둘 모두 센서별 intrinsic 보정의 대용품은 아니다.
- `build_sweepick_measured_urdf.py`는 기존 STL reference만 mm scale .001로 교체한다. 후속 `build_sweepick_measured_unit_urdf.py`는 m로 구운 unit STL을 만들고 measured config가 이 버전을 사용한다. 실제 generated measured STL/URDF는 Git에 없다. 원본 `so101_pgripper.urdf`나 calibration planner URDF는 새 base로 대체하지 않는다.
- `envs/sweepick_block40_measured.py`: table bias를 arms/static camera에 적용하고, 저장 pointcloud를 World→base로 변환한다. 기존 housing proxy는 실측 확정체가 아니다.
- `envs/robot/planner.py`: CuRobo `world_config`에는 table만 있다. measured `curobo.yml`은 별도 `so101_pgripper_calib.urdf`, collision spheres 설정은 null이다. chassis, rails, 새 mount, 반대 팔 간 충돌을 검증한 상태가 아니다.

## 2026-10-20 실제 SDK 자료 수령 후

실제 데이터의 K, depth 계약과 두 경험적 extrinsic을 수령했다. optical TF 포함 비교 URDF와 exact K 어댑터는 `received_20261020/` 및 `SWEEPICK_OS30A_RECEIVED_20261020.md`를 참고한다. 기존 mechanical/preview 모델은 광학값을 자동 주입하지 않고 그대로 보존한다. 자료에는 미승인 candidate가 포함되며 절대 보정 완료 상태가 아니다.

## 광학정보 수령 전: 기계 URDF 먼저 사용

후속 사용자 요청에 따라 `--mechanical-only` 경로를 추가했다. 준비된 파일은 `assets/sweepick_camera_candidate/mechanical/sweepick_mechanical.urdf`와 인접한 `meshes/`다. root는 base_footprint이며 두 mount와 OS30A 바디 proxy를 포함한다. 실제 TurtleBot/양팔은 로컬 팀 원본을 아래 명령으로 합친다.

```bash
python tools/sweepick_camera_candidate/build.py --mechanical-only \
  --robot-urdf /absolute/path/team-expanded.urdf \
  --arm-urdf /absolute/path/so101_pgripper_sweepick_unit.urdf \
  --out /tmp/sweepick-mechanical-01
```

팀 robot URDF에 양팔이 이미 있으면 `--arm-urdf`는 생략한다. package URI가 있으면 기존 안내대로 `--package-dir`를 추가한다. 생성파일 이름은 `sweepick_mechanical.urdf`다. `--mechanical-only`는 calibration JSON에 광학값이 채워져 있어도 optical TF나 SAPIEN camera config를 만들지 않는다. 기존 작동 RoboTwin 카메라 설정은 이 모드로 바뀌지 않는다.

지금 넣는 `camera_link`는 카메라 바디의 체결 기준이며 실제 촬영 원점이 아니다. 바디 방향은 CAD 후보의 하향 65도를 유지하며 사용자 시선 하향 25도와의 차이는 미확정 상태다. 두 mount의 visual/collision에는 CAD 메시를, 카메라 바디에는 치수 기반 box proxy를 쓴다. 공칭 질량을 임의 분배한 inertia는 추가하지 않았고 정적 마운트용이다. 실제 SAPIEN/CuRobo 접촉/충돌/동역학 검증은 별도다.

광학자료 수령 후 같은 기계 장착 변환에 body→optical 보정을 연결하고 `--mechanical-only` 없이 새 출력 폴더로 생성한다. URDF optical TF와 RoboTwin static camera 설정은 같은 변환에서 함께 만들므로 위치를 따로 손으로 맞추지 않는다. 아직 남은 기계 조립 방향/각도 차이도 그 전에 확인해야 한다.

## 재현/사용

작동 중인 checkout을 바꾸지 말고 새 경로에 새 브랜치를 clone한다. 필요 라이브러리는 도구 폴더의 requirements를 별도 venv에 설치한다. CAD 재생성 외에는 cadquery가 필요하지 않다.

```bash
git clone --single-branch --branch codex/sweepick-camera-cad-candidate-20261020 \
  https://github.com/jtjin0916/RoboTwin.git RoboTwin-camera-candidate
cd RoboTwin-camera-candidate
python3 -m venv .venv-camera
. .venv-camera/bin/activate
python -m pip install -r tools/sweepick_camera_candidate/requirements.txt
python tools/sweepick_camera_candidate/cad_audit.py
python -m pytest -q tests/test_sweepick_camera_candidate.py
python tools/sweepick_camera_candidate/build.py --out /tmp/sweepick-camera-preview-01
python tools/sweepick_camera_candidate/render_preview.py --out /tmp/sweepick-camera-preview-01/preview.png
```

`--out` 경로가 이미 있으면 실패한다. 입력 원본을 수정하지 않는다. 생성 standalone URDF의 root는 base_footprint다. mesh 3개 모두 복사되므로 폴더째 이동 가능하다. 실제 ROS2 source xacro는 먼저 팀 환경에서 `xacro /absolute/path/robot.urdf.xacro > /tmp/team-expanded.urdf`로 확장한다. 팀 모델에 이미 두 팔이 있으면 `--arm-urdf`를 생략한다.

```bash
python tools/sweepick_camera_candidate/build.py \
  --robot-urdf /tmp/team-expanded.urdf \
  --package-dir turtlebot3_description=/absolute/path/turtlebot3_description \
  --arm-urdf /absolute/path/so101_pgripper_sweepick_unit.urdf \
  --out /tmp/sweepick-combined-candidate-01
check_urdf /tmp/sweepick-combined-candidate-01/sweepick_candidate.urdf
```

위 경로는 팀 로컬 파일로 바꿔야 한다. 실제 TurtleBot 없는 상태에서 대체 chassis를 만들어 통합 완료로 주장하지 않는다. `--arm-urdf`는 좌/우 prefix를 붙여 두 번 삽입한다. link/joint/material/transmission을 보존하며 알 수 없는 arm plugin/extension은 거부한다. package URI는 명시적 package-dir로 해석한다. ROS2 controller 파일은 이 작업에서 자동 수정하지 않으며 prefix joint 이름과 controller joint 목록을 팀이 맞춰야 한다. 기존 chassis나 arm에 camera frame이 있으면 충돌을 감지하고 중단한다.

Xacro를 선호하면 `tools/sweepick_camera_candidate/camera_mount.urdf.xacro`를 include한 뒤 `sweepick_camera_mount_candidate parent="base_footprint" prefix="" mesh_dir="..."`를 호출한다. 매크로는 mechanical candidate만 만들며 optical TF는 없다. 생성 URDF와 Xacro의 TF가 같음을 CPU 테스트했다.

실측 뒤 `calibration.template.json`을 별도 복사해 optical_xyz (camera_link의 m), SDK 기준 optical_rpy, assembly yaw/seating_delta, 실측자와 증거를 기록한다. 검토 후 status를 `measured_and_reviewed`로 바꾼다. JSON에 숫자를 채웠다는 사실 자체가 물리적 보정 검증은 아니다. `--calibration /path/calibration.json`을 build.py에 주면 optical TF와 별도 SAPIEN config도 생성한다. 숫자를 넣은 미검증 preview는 status=unverified를 유지한다. calibration 없이 optical TF가 없는 것은 의도된 동작이다.

팀 SAPIEN 3 렌더링 환경에서 stationary viewer:

```bash
python tools/sweepick_camera_candidate/preview_sapien.py \
  --model /tmp/sweepick-camera-preview-01/sweepick_candidate.urdf
```

이는 geometry-only viewer로 arm motion이나 데이터 수집을 시작하지 않는다. CPU에서 SAPIEN renderer를 실행/검증하지 못했다. 실제 센서 extrinsic이 준비되면 `--calibration ...`을 추가한다. 미검증값의 카메라 preview는 `--allow-unverified-camera`를 명시한다. 기존 measured task에 적용하려면 별도 task를 복사한 다음 `runtime.attach_mount_preview`를 사용하고, 기존 가짜 housing proxy와 중복 TF publisher를 제거한다. static head config는 `model.sapien_camera()` 출력으로 교체하며 World 변환/table bias를 중복 적용하지 않는다. 이 작업은 로컬 실제 로봇 파일을 확보한 뒤 해야 한다.

Depth-only 제약: head RGB collection을 끄고 enable_color_stream=false, collect_wrist_camera=false로 head만 먼저 확인한다. 기존 `Camera.get_pcd()`는 synthetic RGB 열도 붙이므로 `runtime.depth_only_observation()`으로 XYZ만 남기는 별도 preview 경로를 사용한다. 실제 depth 학습 데이터에 렌더 RGB 또는 colorized depth를 RGB 카메라 정보처럼 섞지 않는다. 기존 ACT가 RGB 입력을 요구하면 별도 depth encoder/입력규격 변경이 필요하며 이 패키지는 기존 RGB ACT 호환성을 주장하지 않는다. 실제 CameraInfo의 K/D/R/P, depth unit, valid mask, reference frame, 시간 동기화를 확인해야 한다.

## CuRobo 충돌 인수인계

`curobo_mount_obstacles_CANDIDATE.json`은 각 arm base 좌표계의 padding 5mm AABB 후보다. body와 두 mount의 모든 mesh vertex/box를 포함한다. 슬롯을 채우는 보수적 근사이며 조립이 바뀌면 `runtime.conservative_obstacles(new_urdf)`로 재생성한다. 좌 arm root와 우 arm root가 다르므로 같은 cuboid pose를 양쪽에 복사하면 안 된다. quaternion은 wxyz다.

기존 planner가 읽지 않는 별도 파일이므로 이 JSON을 저장하는 것만으로 충돌 보호가 활성화되지 않는다. 새 planner 실험 복사본에서 기존 table을 유지하고, 각 side의 `arm_frame_obstacles[side]['cuboid']`를 `world_config['cuboid']`에 추가한 후 두 MotionGen 인스턴스를 모두 생성/갱신한다. 다음을 통과해야 motion용으로 승격할 수 있다: 실제 chassis/rails/cables/반대 팔 world obstacles, measured arm geometry와 collision spheres, home과 전체 궤적 swept collision, signed distance/clearance 검증, 중복 self collision 처리. 현재 CPU 검사는 좌표/포함성만 검증했으며 GPU CuRobo/SAPIEN의 접촉/계획 성공이나 물리 안전을 검증하지 않았다.

## 남은 확인 및 인수 조건

1. **확인 완료:** 아래판 112mm 방향은 로봇 전후 X축. **사진 대기:** 위 부품 yaw와 실제 삽입 끝면/깊이 및 회전. 사용자가 조립 사진을 다시 제공할 예정이다.
2. **확인 완료:** 사용자가 지정한 각도는 수평 전방에서 하향 25도. 현재 CAD 후보는 하향 65도여서 불일치하며 조립 사진으로 해소해야 한다.
3. **확인 완료:** OS30A는 후면 M3 홀로 직접 체결. 사용할 플레이트 면과 USB/렌즈 방향 및 screw 체결 길이/케이블 여유는 사진/실물로 확인해야 한다.
4. **팀원 자료 대기:** 바디 rear datum→SDK depth optical frame의 실측 3D 위치/회전, 좌/우 IR 기준과 rectification. 제조사 calibration/CameraInfo 또는 보정 타겟을 이용한 hand-eye/extrinsic 측정이 필요하다.
5. 팀 실제 ROS2 source와 mesh, 기존 generated measured arm unit STL/URDF. 이를 넣은 combined 모델에서 ROS check_urdf/RViz, SAPIEN 정적 로드/시야, optical TF 투영, 양팔 home/작업경로 및 CuRobo 충돌 인수검사.

이번 CPU 검증: 신규 10개 + 기존 관련 48개, **58 passed**. STEP BRep valid, 제안 조립 침범 0, STL watertight와 치수/체적, TF 광학축 부호, World/table 변환, 메시 휴대성/양팔 prefix, Xacro parity, 파일 덮어쓰기 거부, optical null 실패, per-arm AABB 포함성, synthetic RGB 제거를 확인했다. 별도 기존 `test_sweepick_contract.py`는 torch 미설치로 collection 실패하여 이 58개에는 포함하지 않았다. ROS2 check_urdf, 실제 로봇 전체 모델, GPU SAPIEN/CuRobo, 하드웨어 보정은 미실행이다.

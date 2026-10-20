# OS30A 수령 자료 검토 및 시뮬레이션 비교 설정 (2026-10-20)

자료: `sweepick_261020_os30a_data.zip`. 원본은 수정하지 않았다. ZIP 내 문서/소스는 분석 자료로만 취급했고 SDK helper를 실행하지 않았다. 기존 측정 task, arm 설정, 카메라 설정, 수집 데이터는 변경하지 않는다. 숫자 기반 프로필과 재현 도구만 별도 추가했다. 원본 영상과 전체 로그는 Git에 올리지 않았다.

## 이번에 가능한 것

실제 depth와 intrinsic을 읽고, 제공된 base→optical 변환으로 점군을 복원하며, 같은 변환을 재현하는 optical TF 포함 URDF와 SAPIEN 비교 설정을 만들 수 있다. 바디 기준 광학 원점을 별도로 실측하지 않았어도 **경험적으로 얻은 base→optical 변환을 재현하는 시뮬레이션**은 가능하다. 이것은 절대 보정이나 실제 체결면 기준 광학 중심을 확정했다는 뜻은 아니다.

## 직접 확인한 데이터

- `depth_000006.png`: 640×460 uint16. 0이 109,963개, 16384가 7개. 두 코드를 제외한 184,430개 점을 복원했다. 16384의 SDK 의미는 불명확하므로 임시 invalid mask로 처리한다.
- PNG를 little-endian uint16 byte 배열로 복원한 FNV1a64=`1b804e8ead724ba8`은 SDK trace record 6 / serial 11과 정확히 일치한다. 파일이 기록된 해당 ZD depth와 동일한 수치임을 확인한 것이며, 단위/절대 정확도까지 독립 검증한 것은 아니다.
- 원시값 단위는 제공된 코드/보고서의 mm 계약을 따른다. `Z_m=raw*0.001`과 팀의 경험 보정 `Z_m=raw*0.001/1.016`을 구분한다. 1/1.016은 별도 보드 실험에서 얻었다는 제공자 보고이며 이 ZIP에 그 실험 원본은 없다. 두 ICP TF 모두 보정된 depth로 맞춰졌으므로 재현 시에는 같은 보정을 사용한다.
- `depth_mm.npy`: float32 460×640, **30장 중앙값 관측**. 소수 화소가 17,146개이며 원본 단일 프레임으로 취급하지 않는다. SCENE.json은 이 관측에 속하고 frame 6과 동일 시각이 아니다.
- top.png(1280×920), top_000006.jpg(640×460)의 RGB 세 채널은 모든 화소에서 동일하다. projector 무늬가 보이며, IR 영상이라는 제공자 설명과 일치한다. 실제 RGB 카메라 정보로 쓰지 않는다. 정류된 왼쪽 IR이라는 세부 정체는 제공자의 판단이고 전체 factory log/SDK 점군으로 추가 확인해야 한다.
- helper는 `initStreamFS`, color MJPG 1280×920/15, depth scale-down 11-bit 640×460, `IMAGE_SN_SYNC`, rectify index 0을 사용한다. stderr에는 depth 장치 30fps 설정이 있다. SDK trace 6,692개 depth timestamp의 223.526초 구간에서 전달 FrameSet 평균은 약 **29.934Hz**다. color exposure FPS나 host latency를 독립 측정한 값은 아니다.

원본별 SHA256, 추출 근거, 수치 검사는 `assets/sweepick_camera_candidate/received_20261020/audit.json`에 기록했다. SDK stderr의 버전 문자열만으로 firmware 버전을 확정하지 않았다.

## 실제 intrinsic

SCENE.json의 reference K (1280×920):

```
fx=fy=818.5999755859375
cx=626.2974853515625
cy=462.41552734375
```

제공된 코드의 pixel 대응 `u_ref=2*u_depth`, `v_ref=2*v_depth`, 반 화소 보정 없음에 따라 depth K (640×460)는 다음과 같다.

```
[409.29998779296875, 0, 313.14874267578125]
[0, 409.29998779296875, 231.207763671875]
[0, 0, 1]
```

이는 팀 코드 재현값이다. full factory R/T/D와 원본 rectify log가 없으므로 이 픽셀 대응 자체를 제조사 calibration으로 독립 확인한 것은 아니다. 정류 영상의 D=0 처리는 제공자 가정이며 raw distortion을 측정한 값으로 저장하지 않았다. 영상 경계까지의 공칭 각도로 환산하면 수평 약76.03°, 수직58.67°다. fovy만 58.7°로 맞추면 중심점 (313.149,231.208)이 누락되므로 full K를 적용해야 한다.

SAPIEN 공식 소스의 호출 순서에 맞게 `set_perspective_parameters(near, far, fx, fy, cx, cy, skew)` 어댑터를 제공한다.
https://sapien-sim.github.io/docs/_modules/sapien/sensor/stereodepth.html

## 두 extrinsic의 상태

변환 방향: `p_base = T_base_from_optical @ p_optical_h`. 자료의 4×4 행렬을 기준으로 삼고 반올림된 translation/quaternion 필드로 다시 만들지 않았다. 두 R은 proper orthonormal rotation임을 확인했다.

| 프로필 | optical origin in base (m) | 광축 하향각 | 근거/한계 |
|---|---|---:|---|
| in_use | (-0.0330105671, +0.0220671493, 0.5129377312) | 69.5606° | 제공자가 기존 사용 중이라고 표시. 이전 팔 root (-.0961,±.15,.140)에 맞춘 ICP |
| candidate | (-0.0344237922, +0.0266162712, 0.4861497724) | 65.4957° | 제공자가 미승인·미사용이라고 표시. 새 팔 root (-.060,±.15,.123)에 다시 맞춘 ICP |

candidate는 CAD 체결면 법선 하향65°와 가깝다. 이전 사용자 표현인 수평 전방 하향25°와는 다르다. 시뮬레이션 비교 파일은 제공된 경험값을 그대로 재현하며, 이전 25° 요구가 물리적으로 확인됐다고 바꾸지 않는다. 비교 프로필 선택과 실제 로봇 운용 승격은 구분한다.

검증 P5 surface median 7.6→5.6mm 개선은 **제공자 보고**다. 맞춤 자세 P1~P5의 원시 점군/mesh 입력이 ZIP에 없으므로 재계산하지 못했다. (+27..31,+15)mm tool-minus-block 잔차 원인도 미확정이다.

### 바닥/블록 좌표도 함께 확인해야 한다

SCENE의 `T_world_camera`는 실제로 제공된 in_use base TF와 숫자가 같다. RoboTwin World와 혼동하면 안 된다. RoboTwin은 `R_WB=Rz(+90°)`, `t_WB=(0,-.2539,tabletop_world_z)`를 추가한다.

SCENE에 보고된 같은 블록 중심을 새 TF로 재표현하면 약 `(0.1670,0.0465,0.0223)`→`(0.2000,0.0457,0.0108)`m로 바뀐다. 이는 같은 관측점의 좌표 재표현이며 새 물체 탐지/독립 실측이 아니다. candidate에서 X가 약33mm 이동한다. 같은 보고 바닥면은 그 부근에서 base z≈-9mm가 되어 z=0과 차이가 남는다. 팔 surface ICP 오차 감소만으로 바닥/블록 절대좌표가 맞다고 결론 내릴 수 없다. task crop 안의 raw sample 점 수가 old=7,994, candidate=1,879로 달라지는 것도 바닥 위치와 crop 영향이 섞인 결과이며 품질 순위가 아니다.

## 생성 모델과 실제 바디 보정의 차이

`received_20261020/{in_use,candidate}/sweepick_received.urdf`는 기존 mechanical CAD에 optical TF를 연결한 mount-only 모델이다. 바디→optical은 다음 식으로 **역산한 값**이다.

```
T_body_from_optical = inverse(T_base_from_body_CAD_candidate) @ T_base_from_optical_ICP
```

candidate의 역산 translation은 약 `(26.842,26.429,-10.444)`mm다. 이는 CAD body 후보와 경험 TF를 동시에 재현하기 위한 분해이며, 후면 체결면에서 실측한 광학 중심이 아니다. body geometry 오차/조립 오차와 ICP 오차를 포함할 수 있다. 두 모델 모두 실제 TurtleBot 및 measured arm 파일은 포함하지 않는다. 기존 local source 통합 builder와 별개로 mount-only optical 비교 모델을 제공한 것이다.

## 재현과 적용

```bash
python tools/sweepick_camera_candidate/ingest_received.py \
  --archive /absolute/path/sweepick_261020_os30a_data.zip \
  --out /tmp/os30a-received-review-01
```

새 출력 경로만 허용한다. SDK helper는 실행하지 않는다. 출력에는 두 URDF/메시, profile.json, static head override, audit.json이 생긴다. active task/embodiment registry에는 등록하거나 자동 적용하지 않는다.

실제 measured task를 복사한 **별도 preview task**에서 카메라 생성이 끝난 뒤 다음을 호출한다. 도구 폴더를 import 경로에 추가한다. 프로필은 명시적으로 `candidate` 또는 `in_use`를 선택한다.

```python
# 별도 preview task의 load_camera 메서드
from pathlib import Path
import json
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/sweepick_camera_candidate"))
from received_data import apply_to_measured_task

def load_camera(self, **kwargs):
    super().load_camera(**kwargs)
    profile_path = Path(__file__).resolve().parents[1] / "assets/sweepick_camera_candidate/received_20261020/candidate/profile.json"
    profile = json.loads(profile_path.read_text())
    apply_to_measured_task(self, profile)
    self.scene.update_render()
```

이 함수는 static head_camera 하나에만 exact K와 pose를 설정한다. task의 최종 tabletop_world_z를 사용하므로 table 높이를 두 번 더하지 않는다. 카메라 randomization을 끈 비교용이며 로봇 운동/수집을 자동 시작하지 않는다. 실제 adapter GPU 실행은 이 환경에서 미검증이다. 기존 `static_camera_list` YAML만 복사하면 기본 Camera 클래스가 intrinsic_K를 무시하므로 위 full K adapter가 필요하다.

실제 수신 depth 처리:

```python
from received_data import unproject_depth
optical_xyz, base_xyz, valid = unproject_depth(
    raw_depth_mm, profile, apply_empirical_scale=True
)
```

0/16384/비유한 값을 먼저 제외한다. correction은 호출자가 반드시 선택한다. 이미 보정한 real depth에 다시 적용하지 않는다. SAPIEN 이상적 depth는 물리 길이이므로 이 어댑터에서 **1/1.016을 다시 적용하지 않는다**. 실제 센서 오차를 모사하려면 별도 실험 모델로 분리해야 한다. 렌더 RGB를 OS30A RGB로 기록하지 말고 기존 depth-only observation 처리 경로를 사용한다. collision safety 상태는 이전과 동일하게 미검증이다.

## 추가로 필요한 최소 자료

1. 같은 mode/index의 **factory rectify log 전체 1024-byte 원본 + 필드명 포함 decoded JSON**: CamMat1/2, NewCamMat1/2, D1/D2, R/T, output resolution, 재투영/Q 관련 항목과 구조체/SDK 버전. 다른 모드 보정값과 섞지 않는다.
2. 같은 FrameSet의 **raw uint16 depth + SDK pc_xyz + factory log + 원본 프레임 메타데이터** 1세트. 현재 ZIP에는 pointcloud 지문만 있고 XYZ payload가 없어 역투영과의 수치 일치를 검증할 수 없다. `READ03_NO_PC` 등의 현재 설정과 export 방식도 기록한다. RGB/IR 정합, Z-depth와 range, 단위, half-pixel 규칙, SDK invalid mask 차이를 함께 확인한다.
3. 팔 mesh fitting과 독립된 **평평한 바닥/보정판/위치를 잰 40mm 블록**의 거리·자세 기준. 측정 기준면/원점과 raw data를 함께 제공해 absolute extrinsic, scale1/1.016, 바닥 z=0, tool 기준 오차를 분리한다. 특히 reported +27..31,+15mm 잔차는 원인이 확인될 때까지 카메라 offset에 임의로 더하지 않는다.

바디 기준 광학 원점이 필요하면 별도의 제조사 기계광학 도면 또는 외부 보정으로 얻어야 한다. 다만 현재의 base→optical 경험값을 이용한 시뮬레이션 비교는 위 자료를 기다리지 않고 실행할 수 있다.

## 검증 범위

CPU: 신규 received 데이터 테스트 10개 + 이전 관련58개 = **68 passed**. real archive ingest 성공, raw depth trace 지문 일치, K scaling/pixel convention, invalid/scale 처리, optical 역투영·재투영, URDF full-chain=제공 행렬, World/table 변환, exact K setter 인자 및 head-only adapter를 확인했다. setter는 stub 테스트이며 실제 GPU 렌더링 성공을 의미하지 않는다. P5 재맞춤, ROS2 전체 robot merge, GPU SAPIEN/CuRobo, 하드웨어 보정·충돌 안전은 미실행이다.

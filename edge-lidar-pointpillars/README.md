# Edge LiDAR PointPillars 실습 프로젝트 (개인 포트폴리오)

## 1. 프로젝트 한 줄 요약

**LiDAR 기반 3D 객체 검출(PointPillars)의 전체 파이프라인을 이해하고, Edge 환경에서 성능·지연시간 트레이드오프를 설명할 수 있도록 설계한 개인 실습 프로젝트**

> ✔ GPU 없이 개인 MacBook(CPU)에서 실행 가능
> ✔ 실제 산업 표준 구조(PointPillars) 그대로 반영
> ✔ 추후 Linux + GPU + TensorRT 환경으로 확장 가능

---

## 2. 프로젝트를 만든 이유 (Why)

실무에서 AI 모델은 **"정확하기만 한 코드"**보다,

* 왜 이 구조를 선택했는지
* 어떤 부분이 성능 병목인지
* Edge 디바이스에서 무엇을 포기하고 무엇을 유지해야 하는지

를 **설명할 수 있는 능력**이 더 중요하다.

본 프로젝트는 LiDAR·자율주행 도메인 경험이 없더라도,
**3D 객체 검출 시스템의 구조와 설계, 의사결정 과정을 이해**하기 위해 설계되었다.

---

## 3. 전체 파이프라인 개요

```text
LiDAR Point Cloud (x, y, z, intensity)
        ↓
Pillarization (공간 격자화 + padding)
        ↓
Pillar Encoder (PointNet 방식)
        ↓
BEV Feature Map (2D)
        ↓
2D CNN Backbone
        ↓
Detection Head
        ↓
Class score map + 3D box regression
```

### 핵심 아이디어

* 3D Point Cloud → **2D BEV 문제로 변환**
* 3D CNN 대신 **2D CNN 사용 → 속도 향상**

---

## 4. 출력 결과의 의미

실행 시 다음과 같은 출력이 생성된다:

```text
Cls shape: [1, 3, 40, 40]
Reg shape: [1, 7, 40, 40]
```

### Cls (Classification)

* 40×40 BEV 공간의 **각 셀마다**
* 3개 클래스(Car / Pedestrian / Cyclist)의 존재 확률 예측

### Reg (Regression)

* 동일한 40×40 공간에서
* 각 위치별 3D Bounding Box 파라미터 예측

| Channel | 의미                    |
| ------- | --------------------- |
| 0~2     | x, y, z 위치            |
| 3~5     | width, length, height |
| 6       | yaw (회전 각도)           |

---

## 5. 구현 시 핵심 기술 포인트

### 5.1 Pillar Padding (실제 구현에서 중요한 부분)

LiDAR 포인트는 밀도가 불균일하여 pillar마다 점 개수가 다르다.

따라서 모든 pillar를 동일한 shape으로 만들기 위해:

* `MAX_POINTS_PER_PILLAR` 기준으로 자르거나
* 부족한 부분은 **0 padding**

을 수행한다.

이는 실제 PointPillars 구현에서도 필수적인 전처리 단계다.

---

### 5.2 FP32 / FP16 개념 실험

본 프로젝트는 CPU 환경이지만, 다음 구조를 그대로 따른다:

| Precision | 의미       | Edge 관점         |
| --------- | -------- | --------------- |
| FP32      | 기본 실수 연산 | 정확하지만 느림        |
| FP16      | 반정밀도     | Edge에서 가장 많이 사용 |
| INT8      | 정수 연산    | 가장 빠르나 보정 필요    |

코드에서 `use_fp16` 플래그를 통해 **정밀도 전환 구조**를 확인할 수 있다.

---

## 6. 개발 환경

### 하드웨어

* 개인 MacBook (CPU only)

### 소프트웨어

* Python 3.9+
* PyTorch (CPU)
* NumPy

### 실행 환경

* Python venv 가상환경 사용

---

## 7. 프로젝트 구조

```text
edge-lidar-pointpillars/
├── main.py              # 실행 진입점
├── config.py            # 설정 파라미터
├── data/                # Fake LiDAR 데이터
├── utils/               # Pillar 생성 로직
└── model/               # Encoder / Backbone / Head
```

---

## 8. 이 프로젝트가 보여주는 역량

* LiDAR / 3D Detection **구조적 이해**
* Edge AI에서의 **성능–정확도 트레이드오프 감각**
* 실험 결과를 **텐서 구조와 출력 의미로 설명 가능**
* GPU 환경으로 확장 가능한 코드 설계

---

## 9. 향후 확장 방향

* Linux + NVIDIA GPU 환경에서 TensorRT 적용
* FP16 / INT8 실제 latency 측정
* Anchor 기반 multi-class detection 확장
* BEV 해상도 변화에 따른 성능 분석

---

## 10. 저자

JM Kim
개인 연구·실습 기반 Edge AI 포트폴리오 프로젝트

---

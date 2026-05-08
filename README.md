# AI Highlight Manager

로컬 경기 영상에서 하이라이트 후보 클립을 자동 추출하고, 웹 UI로 라벨링·학습 데이터를 쌓아 XGBoost 모델을 점진적으로 개선하는 파이프라인입니다.

## 구성

- `extract.py` — YOLO 탐지 + 피처 엔지니어링 + 룰/XGB 앙상블 스코어 → 클립 추출
- `make_dataset.py` — Good/Bad 클립 + `event_marks.json` → 학습 CSV 생성
- `train_model.py` — GroupKFold 교차검증 + XGBoost 학습
- `collect_hard_negatives.py` — 하드 네거티브 자동 마이닝
- `app/server.py` — FastAPI 서버 (추출/라벨링 API)
- `app/static/index.html` — 웹 UI

## 빠른 시작

```bash
# 학습 데이터 수집 모드 (기본, 권장)
python -m uvicorn app.server:app --reload

# 실사용 모드 (Lv3 이후에만)
USE_XGB=1 python -m uvicorn app.server:app --reload
```

브라우저: `http://localhost:8000`
웹 헤더 배지로 현재 모드 확인 가능.

---

## 핵심 원칙

### 🔴 원칙 1: 데이터 수집 중에는 XGB OFF

XGB가 켜진 상태로 추출하면 **모델이 이미 놓치는 유형의 이벤트는 클립 후보로도 안 올라옵니다** (확증 편향 루프). 그 유형은 영원히 학습 데이터에 들어가지 않고 FN이 고착됩니다.

- Lv0 ~ Lv2 내내 `XGB OFF` 유지
- Lv3 도달(20+ 경기, 성능 안정) 후에만 `USE_XGB=1` 로 실사용 시작
- 웹 헤더 배지가 매번 `XGB OFF` 인지 확인

### 🔴 원칙 2: 재학습은 항상 전체 데이터로

XGBoost는 이어학습이 아닙니다. `train_model.py` 실행할 때마다 **누적된 전체 CSV로 처음부터 다시 학습**됩니다. 이전 `.xgb` 파일은 덮어씌워집니다.

중요한 건 모델 파일이 아니라 **데이터 원본**입니다:

- `highlights/Good/`, `highlights/Bad/` — 라벨링된 클립
- `event_marks.json` — 수동 이벤트 마킹
- `runs/` — 각 extract 실행의 xh_scores, detections, clip_mapping

이 셋만 있으면 언제든 데이터셋과 모델을 완전히 재생성할 수 있습니다.

### 🔴 원칙 3: `ai_training_dataset.csv`는 일회용 파생물

`make_dataset.py` 실행 시마다 기본적으로 **덮어쓰기**됩니다. 스키마 변경(피처 추가/제거) 시 오염 방지용.
누적이 필요한 특수 상황(hard negative 추가 등)에만 `--append`.

---

## 단계별 학습 계획

| 단계           | 경기 수 | 예상 샘플 | 주요 목표     | AUC 기대    |
| -------------- | ------- | --------- | ------------- | ----------- |
| **Lv0 (현재)** | 4       | ~720      | 수정사항 검증 | 0.83 ± 0.08 |
| **Lv1**        | 8       | ~1,450    | 첫 실용 모델  | 0.85 ± 0.05 |
| **Lv2**        | 12      | ~2,200    | 일반화 확인   | 0.88 ± 0.03 |
| **Lv3**        | 20+     | ~3,600+   | 프로덕션 투입 | 0.90+ 안정  |

다음 단계 진행 판단: **폴드 분산이 이전 단계 대비 30% 이상 감소했을 때**. 정체되면 데이터 다양성 재점검.

---

## 워크플로우

### A. 첫 학습 준비 (Lv0 → Lv1: 4경기 추가)

#### 1단계. 서버 기동 (XGB OFF 확인)

```bash
python -m uvicorn app.server:app --reload
```

콘솔 출력:

```
[XGB OFF] 학습 데이터 수집 모드 — 룰 기반 스코어만 사용합니다.
```

웹 헤더에 🟠 `XGB OFF` 배지 확인.

#### 2단계. 경기별로 반복 (추가 4경기)

각 경기마다:

1. **영상 업로드** → 서버가 `run_highlight_pipeline` 실행
2. **후보 클립 확인** (보통 40개 생성)
3. **event_mark 수동 지정**:
   - 슈팅/골/역습 시작점 등 핵심 이벤트의 정확한 프레임 지정
   - **다양성 확보**: 골대 안 보이는 역습, 중거리슛, 코너킥 직후, 수비 차단 등 포함
4. **Good/Bad 라벨링** (UI 버튼)
5. **(선택) Trim** 으로 클립 구간 미세 조정

#### 3단계. 데이터셋 재생성 + 학습

```bash
# 누적된 8경기 전체로 재생성 (덮어쓰기)
python make_dataset.py

# 학습 + CV + 에러 분석 CSV 생성
python train_model.py
```

출력 확인 포인트:

- `경기(match_id) 수: 8` — 제대로 카운트되는지
- 폴드 간 AUC 분산이 Lv0 대비 감소했는지
- 피처 중요도에서 죽은 피처 없는지

#### 4단계. (선택) 하드 네거티브 보강

```bash
# 미리보기
python collect_hard_negatives.py --per-run 10 --dry-run

# 반영 (ai_training_dataset.csv 에 누적됨)
python collect_hard_negatives.py --per-run 10

# 재학습
python train_model.py
```

---

### B. 데이터 추가 시 (Lv1 → Lv2 → Lv3)

**매번 동일한 루틴**:

```bash
# 1. XGB OFF 확인 (서버 헤더 배지)
python -m uvicorn app.server:app --reload

# 2. 웹에서 새 경기 영상 업로드 + 라벨링 + event_mark 지정

# 3. 데이터셋 재생성 (기존 덮어쓰기)
python make_dataset.py

# 4. 학습 (전체 데이터로 처음부터)
python train_model.py
```

**주의**: 새 경기 추출 시 이전 `.xgb` 파일이 있어도 `USE_XGB` 환경변수 없으면 자동으로 무시됩니다. 그래도 백업이 불안하면:
ㅋ

```bash
mv highlight_model.xgb highlight_model.lv1.xgb.bak   # 단계별 아카이브
```

---

### C. Lv3 도달 후 실사용 전환

```bash
# 실사용 모드로 서버 기동
USE_XGB=1 python -m uvicorn app.server:app --reload
```

웹 헤더에 🟢 `XGB ON` 배지 확인.

**이 상태로 추출한 클립은 학습 데이터에 넣지 마세요.** 데이터 수집은 별도 세션(XGB OFF)으로 진행.

---

## 데이터 다양성 가이드

성능 정체의 가장 큰 원인은 데이터 편중입니다. 새 경기 수집 시 다음을 의도적으로 포함:

### 포함해야 할 상황

- **골대 화면 밖 이벤트**: 역습 시작, 중거리슛, 빌드업
- **다양한 카메라 각도**: 메인캠/골라인캠/와이드
- **조명 조건**: 주간/야간/그림자
- **경기 템포**: 빠른 역습 vs 느린 빌드업

### 피해야 할 편향

- 같은 경기장만 반복 (카메라 고유 특성 과적합)
- event_mark 없이 algorithm 출처 라벨만 쌓기 (확증 편향 재발)
- Good 클립만 쌓고 Bad 방치 (과대 positive → 오탐 증가)

---

## 스크립트 레퍼런스

### `make_dataset.py`

```bash
python make_dataset.py              # 덮어쓰기 (기본, 권장)
python make_dataset.py --append     # 기존 CSV 에 누적 (특수 상황)
```

### `train_model.py`

```bash
python train_model.py
```

출력물:

- `highlight_model.xgb` — 학습된 모델
- `error_analysis.csv` — OOF 오답 분석 (FN/FP 검토용)
- 콘솔: 폴드별 성능, CV 요약, 피처 중요도

### `collect_hard_negatives.py`

```bash
python collect_hard_negatives.py --per-run 10 --dry-run
python collect_hard_negatives.py --per-run 10
```

기본적으로 기존 CSV에 누적됩니다 (이 스크립트는 추가 목적).

### `extract.py` (CLI 단독 실행)

```bash
# 기본 (XGB OFF)
python extract.py

# XGB 사용 (실사용 시)
USE_XGB=1 python extract.py
```

상단의 `VIDEO_PATH` 변수 수정해 대상 영상 지정.

---

## 개발 메모

- YOLO 모델 경로: `best-8.pt`
- 감지 클래스: player, ball, goal, referee, goalkeeper
- 기본 후보 수: 40개 / 경기
- 클립 길이: anchor 기준 앞 15초 + 뒤 10초
- 클립 간 최소 간격: 21초 (중복 방지)

## XGBoost + SHAP 검증

`detections.csv`, `xh_scores.csv`가 있는 폴더를 지정해 SHAP 분석:

- 스크립트: `shap_analysis.py`
- 기본 입력 폴더: `highlights`
- 기본 출력 폴더: `shap_outputs`

## YouTube 추출 (yt-dlp)

```bash
yt-dlp -f 'bv[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080][ext=mp4]' \
  --downloader aria2c \
  --downloader-args "aria2c:-x 16 -s 16 -k 1M" \
  "https://www.youtube.com/watch?v=Fug0Uvy9XAE"
```

# ai-highlight

# 축구 하이라이트 AI 추출 프로젝트

## 핵심 원칙
- 비전 중심 (오디오 보조)
- XGB 기본 OFF: 데이터 수집 시 확증 편향 루프 방지 (`USE_XGB=1`로만 활성화)
- 피처 추가 시 3파일 동기화 필수: `extract.py` / `make_dataset.py` / `train_model.py`

## 기술 스택
- **YOLO** `best-8.pt` | stride=7, conf=0.15, imgsz=1280 | ByteTrack (실패 시 detect-only)
- **Classes**: player, ball, goal, referee, goalkeeper

## 피처 계산 (XHScoreCalculator)
| Phase | 피처 | 설명 |
|-------|------|------|
| 1 | `f_ball_speed/accel/dir_change`, `player_density`, `inv_dist_centroid_masked` | 공 물리량 + 선수 밀집도 + 골대 역거리 (±2초 grace, 비가시 0.3 penalty) |
| 2 | `f_possession_switches`, `f_sprint_count` | ByteTrack track_id 기반 |
| 3 | `f_ball_to_goal_width_ratio`, `f_goal_bbox_width_norm`, `f_players_near_ball` | 골대폭 정규화 (줌 불변) |

## XH Score (룰)
```
xh_score = (inv_dist_centroid*0.4 + player_density*0.25 + f_ball_accel*0.2
            + f_ball_speed*0.1 + f_players_near_ball*0.05 + bonus) * 0.9
           + f_audio * 0.1
bonus: 공-골대 근방+밀집=0.5, 극근방=0.7
```
**Final**: `xh*0.6 + xgb*0.4` → 3초 이동평균 → find_peaks

## ML Features (13개, 3파일 동기화)
10개 base col × 윈도우 통계(anchor/mean/max/min/std) → elite 13:
`inv_dist_centroid_masked_{max,mean}` / `f_ball_speed_{max,anchor}` / `f_ball_accel_std` / `f_goalpost_visible_mean` / `f_ball_dir_change_{max,mean}` / `f_possession_switches_mean` / `f_sprint_count_max` / `f_ball_to_goal_width_ratio_mean` / `f_goal_bbox_width_norm_mean` / `f_players_near_ball_max`

## 파일 역할
| 파일 | 역할 |
|------|------|
| `extract.py` | 탐지→피처→스코어→클립 생성 |
| `app/server.py` | FastAPI: 업로드/분류/이벤트마킹 |
| `make_dataset.py` | Good/Bad+event_marks→CSV (Good 5샘플±2s, Bad 3샘플±1s) |
| `train_model.py` | GroupKFold(5) by match_id + scale_pos_weight |
| `collect_hard_negatives.py` | 고점수 피크 중 Good 근방 제외 → label=0 |
| `event_marks.json` | 사용자 마킹 이벤트 프레임 레지스트리 (anchor 최우선) |

## 파이프라인
```
1. 분석:  python extract.py  (USE_XGB=0)
2. 분류:  highlights/Good/ or Bad/
3. 마킹:  UI → event_marks.json
4. 데이터: python make_dataset.py
5. HN:    python collect_hard_negatives.py --per-run 10
6. 학습:  python train_model.py
7. 실사용: USE_XGB=1 uvicorn app.server:app --reload
```

## 이벤트 타입
`goal` `shot` `save` `setpiece` `threat` `defense` `unknown` `none`

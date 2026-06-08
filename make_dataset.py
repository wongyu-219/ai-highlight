"""Phase 1 개선판: 이벤트 프레임 JSON 기반 anchor + 다중 샘플링.

핵심 변경점
- `event_marks.json` (중앙 레지스트리)을 최우선 anchor로 사용
- 폴백: 기존 `clip_mapping.csv` 기반 알고리즘 anchor
- Good 클립은 지정된 이벤트 프레임 ±3초 범위에서 **다중 샘플 (5개)** 생성
  → 작은 라벨 데이터셋 증강 + 윈도우 앵커 편향 완화
- Bad 클립은 원 anchor 기준 1개 샘플만 생성 (과대표집 방지)
- 다중 샘플은 `sample_offset_sec` 컬럼에 기록되어 추적 가능
"""
import argparse
import os
import glob
import json
import re

import pandas as pd

# --- 설정 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIGHLIGHTS_DIR = os.path.join(BASE_DIR, "highlights")
GOOD_DIR = os.path.join(HIGHLIGHTS_DIR, "Good")
BAD_DIR = os.path.join(HIGHLIGHTS_DIR, "Bad")
RUNS_DIR = os.path.join(BASE_DIR, "runs")
DATASET_FILE = os.path.join(BASE_DIR, "ai_training_dataset.csv")
EVENT_MARKS_PATH = os.path.join(BASE_DIR, "event_marks.json")

EVENT_WINDOW_ROWS = 25  # anchor 기준 앞/뒤 25행 통계 윈도우

# Good 클립 다중 샘플링 설정 (±3초 범위에서 몇 개의 anchor를 뽑을지)
POS_SAMPLE_OFFSETS_SEC = [-2.0, -1.0, 0.0, 1.0, 2.0]
# Bad 클립 샘플링: pos/neg 비율 불균형 완화를 위해 ±1초 3샘플로 확장
NEG_SAMPLE_OFFSETS_SEC = [-1.0, 0.0, 1.0]

# Option B: 표준 이벤트 타입 (server.py 와 동기화)
VALID_EVENT_TYPES = {
    "goal", "shot", "save", "setpiece",
    "threat", "defense",
    "unknown", "none",
}

# ML 윈도우 통계를 계산할 base 컬럼 (extract.py의 _ML_BASE_COLS와 동기화)
FEATURE_COLS = [
    'inv_dist_centroid_masked',
    'f_ball_speed', 'f_ball_accel',
    'f_goalpost_visible',
    'f_ball_dir_change',
    'f_possession_switches',
    'f_sprint_count',
    'f_ball_to_goal_width_ratio',
    'f_goal_bbox_width_norm',
    'f_players_near_ball',
    'f_ball_visible',
    'f_ball_to_goal_approach',
]

# XGBoost 입력 피처 (extract.py / train_model.py의 ML_FEATURES와 동기화)
ML_FEATURES = [
    "inv_dist_centroid_masked_max",
    "inv_dist_centroid_masked_mean",
    "f_ball_speed_max",
    "f_ball_speed_anchor",
    "f_ball_accel_std",
    "f_goalpost_visible_mean",
    "f_ball_dir_change_max",
    "f_ball_dir_change_mean",
    "f_possession_switches_mean",
    "f_sprint_count_max",
    "f_ball_to_goal_width_ratio_mean",
    "f_goal_bbox_width_norm_mean",
    "f_players_near_ball_max",
    "f_ball_visible_mean",
    "f_ball_to_goal_approach_max",
]


def load_event_marks() -> dict:
    if not os.path.exists(EVENT_MARKS_PATH):
        return {}
    try:
        with open(EVENT_MARKS_PATH, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def build_event_features(scores_df: pd.DataFrame, anchor_frame: int, frame_id: str, extra: dict | None = None) -> dict | None:
    """anchor 프레임 주변 윈도우 통계 기반 피처 1행 생성."""
    if 'frame' not in scores_df.columns or scores_df.empty:
        return None

    sdf = scores_df.sort_values('frame').reset_index(drop=True)
    anchor_idx = (sdf['frame'] - int(anchor_frame)).abs().idxmin()
    anchor_resolved_frame = int(sdf.loc[anchor_idx, 'frame'])

    start_idx = max(0, anchor_idx - EVENT_WINDOW_ROWS)
    end_idx = min(len(sdf), anchor_idx + EVENT_WINDOW_ROWS + 1)
    window_df = sdf.iloc[start_idx:end_idx]

    full_row = {}
    for col in FEATURE_COLS:
        if col in sdf.columns:
            anchor_val = float(sdf.loc[anchor_idx, col])
            series = window_df[col].fillna(0.0)
        else:
            anchor_val = 0.0
            series = pd.Series([0.0])

        full_row[f'{col}_anchor'] = anchor_val
        full_row[f'{col}_mean'] = float(series.mean())
        full_row[f'{col}_max'] = float(series.max())
        full_row[f'{col}_min'] = float(series.min())
        full_row[f'{col}_std'] = float(series.std(ddof=0))

    row = {
        'frame_id': frame_id,
        'anchor_frame': anchor_resolved_frame,
    }
    for feat in ML_FEATURES:
        row[feat] = full_row.get(feat, 0.0)

    if extra:
        row.update(extra)
    return row


def build_multi_sample_features(scores_df: pd.DataFrame, anchor_frame: int, fps: float,
                                 offsets_sec, clip_name: str, run_id: str,
                                 anchor_source: str, label: int,
                                 event_type: str = "none") -> list[dict]:
    """anchor ±offsets(초) 지점들을 anchor로 하여 여러 피처 행을 생성."""
    rows = []
    # event_type 검증
    et = str(event_type).strip().lower() or "none"
    if et not in VALID_EVENT_TYPES:
        et = "unknown" if label == 1 else "none"
    for offset_sec in offsets_sec:
        shifted = int(round(anchor_frame + offset_sec * fps))
        if shifted < 0:
            continue
        frame_id = f"{run_id}_{clip_name}_{shifted}_o{offset_sec:+.1f}"
        extra = {
            'clip_name': clip_name,
            'run_id': run_id,
            'anchor_source': anchor_source,     # 'event_mark' | 'algorithm' | 'manual_name'
            'sample_offset_sec': float(offset_sec),
            'is_highlight': label,
            'event_type': et,
        }
        r = build_event_features(scores_df, shifted, frame_id, extra=extra)
        if r is not None:
            rows.append(r)
    return rows


def init_folders():
    for folder in [GOOD_DIR, BAD_DIR]:
        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
            print(f"폴더 생성됨: {folder}")


def load_all_clip_mappings() -> list[dict]:
    """runs/ & highlights/ 내의 *_clip_mapping.csv + 짝이 맞는 xh_scores.csv 로드."""
    all_mappings = []

    search_dirs = [RUNS_DIR, HIGHLIGHTS_DIR]
    mapping_files = []

    for root_dir in search_dirs:
        if os.path.exists(root_dir):
            mapping_files.extend(glob.glob(os.path.join(root_dir, "**/clip_mapping.csv"), recursive=True))
            mapping_files.extend(glob.glob(os.path.join(root_dir, "**/*_clip_mapping.csv"), recursive=True))
            mapping_files.extend(glob.glob(os.path.join(root_dir, "clip_mapping.csv")))
            mapping_files.extend(glob.glob(os.path.join(root_dir, "*_clip_mapping.csv")))

    mapping_files = list(set(mapping_files))

    for mf in mapping_files:
        run_dir = os.path.dirname(mf)
        mf_name = os.path.basename(mf)
        prefix = mf_name.replace("_clip_mapping.csv", "")
        scores_file = os.path.join(run_dir, f"{prefix}_xh_scores.csv")
        if not os.path.exists(scores_file):
            scores_file = os.path.join(run_dir, "xh_scores.csv")
        if not os.path.exists(scores_file):
            continue

        try:
            m_df = pd.read_csv(mf)
            s_df = pd.read_csv(scores_file)

            path_parts = mf.split(os.sep)
            if RUNS_DIR in path_parts or "runs" in path_parts:
                try:
                    idx = path_parts.index("runs")
                    run_id = path_parts[idx + 1] if len(path_parts) > idx + 1 else prefix
                except ValueError:
                    run_id = prefix
            else:
                run_id = prefix

            for _, row in m_df.iterrows():
                all_mappings.append({
                    'clip_path': row['clip_path'],
                    'frame': int(row['frame']),
                    'run_id': run_id,
                    'scores_df': s_df,
                })
        except Exception as e:
            print(f"파일 읽기 오류 ({mf}): {e}")

    return all_mappings


def find_xh_scores_for_manual(prefix: str):
    """수동 추가 클립: 접두어로 xh_scores.csv 찾기."""
    for root_dir in [RUNS_DIR, HIGHLIGHTS_DIR]:
        if not os.path.exists(root_dir):
            continue
        candidates = glob.glob(os.path.join(root_dir, f"**/{prefix}_xh_scores.csv"), recursive=True)
        if not candidates:
            candidates = glob.glob(os.path.join(root_dir, f"{prefix}_xh_scores.csv"))
        if candidates:
            try:
                return pd.read_csv(candidates[0])
            except Exception:
                pass
    return None


def resolve_clip_anchor(clip_name: str, all_mappings: list[dict], event_marks: dict):
    """클립의 (anchor_frame, fps, scores_df, run_id, source, event_type) 결정.

    우선순위: event_marks.json > clip_mapping.csv > 파일명의 _manual_{frame} 패턴
    event_type 은 event_marks.json 에 있을 때만 반영(없으면 빈 문자열).
    """
    # 1) event_marks.json 최우선
    mark = event_marks.get(clip_name)
    if mark is not None:
        mark_event_type = str(mark.get('event_type', '') or '').strip().lower()
        # scores_df는 mapping에서 찾아야 함
        for m in all_mappings:
            if m['clip_path'] == clip_name:
                return (int(mark['event_frame_original']),
                        float(mark.get('fps', 30.0)),
                        m['scores_df'],
                        m['run_id'],
                        'event_mark',
                        mark_event_type)
        # mapping에 없으면 job_id + prefix로 xh_scores 탐색
        job_id = mark.get('job_id', '')
        scores_df = find_xh_scores_for_manual(job_id) if job_id else None
        if scores_df is not None:
            return (int(mark['event_frame_original']),
                    float(mark.get('fps', 30.0)),
                    scores_df,
                    job_id,
                    'event_mark',
                    mark_event_type)

    # 2) 알고리즘 mapping
    for m in all_mappings:
        if m['clip_path'] == clip_name:
            return (int(m['frame']), 30.0, m['scores_df'], m['run_id'], 'algorithm', '')

    # 3) 수동 파일명 패턴 (_manual_{frame}.mp4)
    manual_match = re.search(r'(.+)_manual_(\d+)\.mp4$', clip_name)
    if manual_match:
        prefix = manual_match.group(1)
        frame = int(manual_match.group(2))
        scores_df = find_xh_scores_for_manual(prefix)
        if scores_df is not None:
            return frame, 30.0, scores_df, prefix, 'manual_name', ''

    return None


def make_dataset(append: bool = False):
    init_folders()

    print("이벤트 마크 레지스트리 로드 중...")
    event_marks = load_event_marks()
    print(f"  event_marks.json 항목 수: {len(event_marks)}")

    print("메타데이터 스캔 중 (runs/ & highlights/)...")
    all_mappings = load_all_clip_mappings()

    good_clips = [os.path.basename(f) for f in glob.glob(os.path.join(GOOD_DIR, "*.mp4"))]
    bad_clips = [os.path.basename(f) for f in glob.glob(os.path.join(BAD_DIR, "*.mp4"))]

    if not good_clips and not bad_clips:
        print(f"분류된 영상이 없습니다. {GOOD_DIR} 또는 {BAD_DIR} 폴더에 하이라이트 영상을 넣어주세요.")
        return

    print(f"분류 현황 - Good: {len(good_clips)}, Bad: {len(bad_clips)}")

    labeled_rows = []
    stats = {
        'good_event_mark': 0, 'good_algorithm': 0, 'good_missing': 0,
        'bad_event_mark': 0, 'bad_algorithm': 0, 'bad_missing': 0,
    }
    event_type_counts: dict[str, int] = {}

    # --- Good 처리: 이벤트 프레임(있으면) ±3s 다중 샘플링 ---
    for clip in good_clips:
        resolved = resolve_clip_anchor(clip, all_mappings, event_marks)
        if resolved is None:
            stats['good_missing'] += 1
            print(f"Warning (Good): {clip} 의 메타데이터를 찾을 수 없습니다.")
            continue
        anchor_frame, fps, s_df, run_id, source, mark_type = resolved
        # Good 인데 event_type 없으면 'unknown' (추후 라벨링 필요)
        et = mark_type if mark_type else 'unknown'
        rows = build_multi_sample_features(
            s_df, anchor_frame, fps, POS_SAMPLE_OFFSETS_SEC,
            clip, run_id, source, label=1, event_type=et,
        )
        labeled_rows.extend(rows)
        stats[f'good_{source}'] = stats.get(f'good_{source}', 0) + 1
        event_type_counts[et] = event_type_counts.get(et, 0) + len(rows)

    # --- Bad 처리: 단일 anchor ---
    for clip in bad_clips:
        resolved = resolve_clip_anchor(clip, all_mappings, event_marks)
        if resolved is None:
            stats['bad_missing'] += 1
            print(f"Warning (Bad): {clip} 의 메타데이터를 찾을 수 없습니다.")
            continue
        anchor_frame, fps, s_df, run_id, source, _ = resolved
        rows = build_multi_sample_features(
            s_df, anchor_frame, fps, NEG_SAMPLE_OFFSETS_SEC,
            clip, run_id, source, label=0, event_type='none',
        )
        labeled_rows.extend(rows)
        stats[f'bad_{source}'] = stats.get(f'bad_{source}', 0) + 1
        event_type_counts['none'] = event_type_counts.get('none', 0) + len(rows)

    if not labeled_rows:
        print("매칭되는 데이터를 생성하지 못했습니다.")
        return

    new_df = pd.DataFrame(labeled_rows)

    # 기본: 덮어쓰기 (스키마 변경 시 오염 방지). --append 시에만 병합.
    if append and os.path.exists(DATASET_FILE):
        existing_df = pd.read_csv(DATASET_FILE)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True, sort=False)
        print(f"[--append] 기존 데이터와 병합 ({len(existing_df)} -> {len(combined_df)})")
    else:
        if os.path.exists(DATASET_FILE):
            print(f"기존 CSV 덮어쓰기 (--append 지정 시 누적 가능)")
        combined_df = new_df
        print(f"데이터셋 생성 ({len(combined_df)}개)")

    # frame_id 기준 dedup (뒤쪽 값 유지 → 가장 최근 빌드 우선)
    final_df = combined_df.drop_duplicates(subset=['frame_id'], keep='last')
    removed = len(combined_df) - len(final_df)
    if removed > 0:
        print(f"중복 데이터 {removed}개 제거됨.")

    # event_type 누락 행 보정:
    #   - is_highlight==1 이면 'unknown' (차후 UI 에서 라벨링 필요)
    #   - is_highlight==0 이면 'none'
    if 'event_type' not in final_df.columns:
        final_df['event_type'] = ''
    if 'is_highlight' in final_df.columns:
        mask_missing = final_df['event_type'].isna() | (final_df['event_type'].astype(str).str.strip() == '')
        final_df.loc[mask_missing & (final_df['is_highlight'] == 1), 'event_type'] = 'unknown'
        final_df.loc[mask_missing & (final_df['is_highlight'] == 0), 'event_type'] = 'none'
    final_df['event_type'] = final_df['event_type'].astype(str).str.strip().str.lower()

    final_df.to_csv(DATASET_FILE, index=False)

    print("\n--- 데이터셋 생성 요약 ---")
    print(f"최종 데이터셋: {DATASET_FILE} (총 {len(final_df)}행)")
    n_pos = int((final_df['is_highlight'] == 1).sum()) if 'is_highlight' in final_df.columns else 0
    n_neg = int((final_df['is_highlight'] == 0).sum()) if 'is_highlight' in final_df.columns else 0
    print(f"  Positive(1): {n_pos}, Negative(0): {n_neg}")
    print(f"  Good   - event_mark: {stats['good_event_mark']}, algorithm: {stats['good_algorithm']}, missing: {stats['good_missing']}")
    print(f"  Bad    - event_mark: {stats['bad_event_mark']}, algorithm: {stats['bad_algorithm']}, missing: {stats['bad_missing']}")
    print(f"  ※ event_mark 비율이 높을수록 라벨 품질이 높음")

    # event_type 분포 (positive 만)
    if 'event_type' in final_df.columns:
        pos_mask = final_df.get('is_highlight', pd.Series([0]*len(final_df))) == 1
        pos_types = final_df.loc[pos_mask, 'event_type'].value_counts()
        print("\n--- event_type 분포 (positive) ---")
        if len(pos_types) == 0:
            print("  (positive 행 없음)")
        else:
            for t, n in pos_types.items():
                print(f"  {t:10s}: {n}")
        unk = int((final_df.loc[pos_mask, 'event_type'] == 'unknown').sum())
        if unk:
            print(f"  ※ 'unknown' {unk}개 — UI 에서 이벤트 타입을 지정하면 보조 분류기가 학습됩니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Good/Bad 클립 + event_marks 기반 학습 데이터셋 생성")
    parser.add_argument('--append', action='store_true',
                        help='기존 ai_training_dataset.csv 에 병합 (기본은 덮어쓰기)')
    args = parser.parse_args()
    make_dataset(append=args.append)

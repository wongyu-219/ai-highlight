"""Hard Negative Mining: 알고리즘이 높은 점수를 매겼지만 실제 이벤트가 아닌 anchor 찾기.

목적
- XGBoost 모델이 '경계 케이스'를 더 잘 학습하도록 hard negative 를 자동 수집
- Good 이벤트와 시간적으로 떨어진(또는 Bad 클립으로 분류된) 고점수 peak 를 negative 로 추가

동작
1. 모든 runs 의 xh_scores.csv 를 스캔하여 smoothed_score 상위 피크 후보 추출
2. Good 클립의 event_frame (없으면 anchor_frame) ±15초 이내는 이벤트 근방이므로 제외
3. 남은 peak 들을 hard negative 로 피처 추출 (label=0) → ai_training_dataset.csv 에 추가

사용법
  python collect_hard_negatives.py --per-run 10 --dry-run
  python collect_hard_negatives.py --per-run 10
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from make_dataset import (  # 동일 로직 재사용
    FEATURE_COLS,
    ML_FEATURES,
    EVENT_WINDOW_ROWS,
    DATASET_FILE,
    EVENT_MARKS_PATH,
    HIGHLIGHTS_DIR,
    RUNS_DIR,
    build_event_features,
    load_all_clip_mappings,
    load_event_marks,
)

EXCLUSION_SEC = 15.0   # Good 이벤트 ±N초 이내는 positive 영역 → 제외
PEAK_DISTANCE_FRAMES = 60  # peak 간 최소 간격 (약 2초 @ 30fps)


def get_job_positive_frames(job_id: str, event_marks: dict,
                             all_mappings: list[dict]) -> list[int]:
    """해당 job(=run_id) 의 Good 이벤트 프레임들 반환 (event_marks 우선)."""
    # Good 폴더 내 클립들에 해당하는 것만 대상
    good_files = {os.path.basename(p) for p in glob.glob(os.path.join(HIGHLIGHTS_DIR, 'Good', '*.mp4'))}
    frames: list[int] = []

    for clip, mark in event_marks.items():
        if mark.get('job_id') == job_id and clip in good_files:
            frames.append(int(mark.get('event_frame_original', 0)))

    # event_mark 가 없는 Good 클립은 알고리즘 anchor 사용
    for m in all_mappings:
        if m['run_id'] != job_id:
            continue
        if m['clip_path'] in good_files and m['clip_path'] not in event_marks:
            frames.append(int(m['frame']))

    return sorted(set(frames))


def find_xh_score_files() -> list[tuple[str, str]]:
    """(run_id, scores_csv_path) 리스트."""
    out = []
    for root in [RUNS_DIR, HIGHLIGHTS_DIR]:
        if not os.path.exists(root):
            continue
        # runs/{job_id}/*_xh_scores.csv
        for p in glob.glob(os.path.join(root, '**', '*_xh_scores.csv'), recursive=True):
            # run_id 추정
            parts = p.split(os.sep)
            try:
                idx = parts.index('runs')
                run_id = parts[idx + 1] if len(parts) > idx + 1 else os.path.splitext(os.path.basename(p))[0]
            except ValueError:
                run_id = os.path.basename(p).replace('_xh_scores.csv', '')
            out.append((run_id, p))
    return out


def mine_hard_negatives(per_run: int, fps_default: float = 30.0):
    event_marks = load_event_marks()
    all_mappings = load_all_clip_mappings()
    score_files = find_xh_score_files()
    print(f"스캔 대상 xh_scores.csv 파일: {len(score_files)}")

    mined_rows = []
    for run_id, scores_path in score_files:
        try:
            df = pd.read_csv(scores_path)
        except Exception as e:
            print(f"  [skip] {scores_path}: {e}")
            continue
        if 'smoothed_score' not in df.columns or 'frame' not in df.columns:
            continue
        df = df.sort_values('frame').reset_index(drop=True)

        # 피크 탐색
        scores = df['smoothed_score'].values
        peaks, props = find_peaks(scores, distance=PEAK_DISTANCE_FRAMES)
        if len(peaks) == 0:
            continue

        # 상위 peak 정렬
        order = np.argsort(-scores[peaks])
        peaks = peaks[order]

        # 해당 run의 positive 프레임 근방 제외
        pos_frames = get_job_positive_frames(run_id, event_marks, all_mappings)
        exclusion_frames = int(EXCLUSION_SEC * fps_default)

        added = 0
        for p in peaks:
            if added >= per_run:
                break
            peak_frame = int(df.loc[p, 'frame'])
            # Good 이벤트 근방이면 스킵
            if any(abs(peak_frame - pf) < exclusion_frames for pf in pos_frames):
                continue

            frame_id = f"hard_neg_{run_id}_{peak_frame}"
            extra = {
                'clip_name': f"{run_id}__hardneg.mp4",
                'run_id': run_id,
                'anchor_source': 'hard_negative',
                'sample_offset_sec': 0.0,
                'is_highlight': 0,
                'event_type': 'none',
            }
            row = build_event_features(df, peak_frame, frame_id, extra=extra)
            if row is not None:
                mined_rows.append(row)
                added += 1

        if added:
            print(f"  {run_id}: {added} hard negatives (excluded pos={len(pos_frames)})")

    return mined_rows


def merge_into_dataset(new_rows: list[dict], dry_run: bool):
    if not new_rows:
        print("수집된 hard negative 없음.")
        return
    new_df = pd.DataFrame(new_rows)
    print(f"\n수집된 hard negative: {len(new_df)}행")

    if dry_run:
        print("(--dry-run) ai_training_dataset.csv 에 쓰지 않음. 샘플:")
        print(new_df.head())
        return

    if os.path.exists(DATASET_FILE):
        existing = pd.read_csv(DATASET_FILE)
        combined = pd.concat([existing, new_df], ignore_index=True, sort=False)
        combined = combined.drop_duplicates(subset=['frame_id'], keep='last')
        before = len(existing)
        combined.to_csv(DATASET_FILE, index=False)
        print(f"{DATASET_FILE} 갱신: {before} → {len(combined)}")
    else:
        new_df.to_csv(DATASET_FILE, index=False)
        print(f"{DATASET_FILE} 새로 생성 ({len(new_df)}행)")


def main():
    parser = argparse.ArgumentParser(description='Hard Negative Mining')
    parser.add_argument('--per-run', type=int, default=10,
                        help='한 경기당 수집할 hard negative 최대 개수')
    parser.add_argument('--dry-run', action='store_true',
                        help='파일에 쓰지 않고 결과만 출력')
    args = parser.parse_args()

    print(f"설정: per_run={args.per_run}, exclusion_sec=±{EXCLUSION_SEC}, dry_run={args.dry_run}")
    rows = mine_hard_negatives(args.per_run)
    merge_into_dataset(rows, args.dry_run)


if __name__ == '__main__':
    main()

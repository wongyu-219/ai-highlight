import cv2
import pandas as pd
import numpy as np
import torch
import xgboost as xgb
from ultralytics import YOLO
from moviepy import VideoFileClip
from scipy.signal import find_peaks
import os
import time
from tqdm import tqdm
from dataclasses import dataclass

# --- 설정 변수 ---
VIDEO_PATH = "test1.mp4"
MODEL_PATH = "best-8.pt"
OUTPUT_DIR = "highlights"
CONF_THRESHOLD = 0.15
SCORE_THRESHOLD = 0.2
CLIP_DURATION_BEFORE = 15
CLIP_DURATION_AFTER = 10
MIN_SECONDS_BETWEEN_CLIPS = 15

# --- XGB 활성화 스위치 ---
# 데이터 수집 기간(Lv1~Lv3)에는 기본 OFF — 확증 편향 방지.
# 실사용/데모 시에만 `USE_XGB=1` 환경변수로 명시적 활성화.
USE_XGB = os.environ.get("USE_XGB", "0") == "1"
AI_HIGHLIGHT_MODEL_PATH = "highlight_model.xgb" if USE_XGB else "highlight_model.DISABLED.xgb"
HIGHLIGHT_COUNT = 40 if USE_XGB else 30  # 실사용: 40, 학습 데이터 수집: 30 (품질 우선)

# --- XGBoost 모델 전역 로드 ---
AI_MODEL = xgb.XGBClassifier()
AI_MODEL_N_FEATURES = 0
if os.path.exists(AI_HIGHLIGHT_MODEL_PATH):
    AI_MODEL.load_model(AI_HIGHLIGHT_MODEL_PATH)
    try:
        AI_MODEL_N_FEATURES = int(AI_MODEL.n_features_in_)
    except AttributeError:
        try:
            AI_MODEL_N_FEATURES = AI_MODEL.get_booster().num_features()
        except Exception:
            AI_MODEL_N_FEATURES = 0
    print(f"[XGB ON] AI 하이라이트 모델 로드: {AI_HIGHLIGHT_MODEL_PATH} (n_features={AI_MODEL_N_FEATURES})")
elif USE_XGB:
    print(f"[경고] USE_XGB=1 이지만 모델 파일이 없습니다: {AI_HIGHLIGHT_MODEL_PATH} → 룰 기반으로 동작합니다.")
else:
    print("[XGB OFF] 학습 데이터 수집 모드 — 룰 기반 스코어만 사용합니다. (실사용 시 `USE_XGB=1` 로 활성화)")

AI_EVENT_WINDOW_ROWS = 25

# ML 윈도우 통계를 계산할 base 컬럼 (make_dataset.py의 FEATURE_COLS와 동기화)
_ML_BASE_COLS = [
    'inv_dist_centroid_masked',
    'f_ball_speed', 'f_ball_accel',
    'f_goalpost_visible',
    'f_ball_dir_change',
    'f_possession_switches',
    'f_sprint_count',
    'f_ball_to_goal_width_ratio',
    'f_goal_bbox_width_norm',
    'f_players_near_ball',
]

# XGBoost 입력 피처 (make_dataset.py / train_model.py의 ML_FEATURES와 동기화)
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
]

XH_WEIGHTS = {
    'goal': 100.0,
    'setpiece': 80.0,
    'base': 1.0
}


@dataclass
class HighlightRunResult:
    success: bool
    message: str
    fps: float = 0.0
    highlight_frames: list[int] | None = None
    output_dir: str = ""
    clip_paths: list[str] | None = None
    clip_features: list[str] | None = None
    clip_feature_stats: dict[int, dict[str, float]] | None = None
    clip_scores: dict[int, float] | None = None  # {frame: smoothed_score}


class XHScoreCalculator:
    def __init__(self, weights: dict, fps: float):
        self.weights = weights
        self.fps = fps

    def _compute_ball_physics(self, df: pd.DataFrame, all_frames: list) -> pd.DataFrame:
        """공 속도·가속도·방향 급변 계산."""
        ball_coords = (df[df['class_name'] == 'ball']
                       .drop_duplicates('frame')
                       .set_index('frame')[['cx', 'cy']])
        ball_stats = pd.DataFrame(index=all_frames)
        ball_stats[['cx', 'cy']] = ball_coords

        ball_interp = ball_stats[['cx', 'cy']].interpolate(method='linear')
        ball_dx = ball_interp['cx'].diff()
        ball_dy = ball_interp['cy'].diff()
        ball_diff = (ball_dx.pow(2) + ball_dy.pow(2)).pow(0.5)

        ball_stats['speed'] = ball_diff.fillna(0)
        ball_stats['accel'] = ball_diff.diff().abs().fillna(0)
        ball_stats['vx'] = ball_dx.fillna(0)
        ball_stats['vy'] = ball_dy.fillna(0)

        v1x, v1y = ball_stats['vx'].shift(1), ball_stats['vy'].shift(1)
        v2x, v2y = ball_stats['vx'], ball_stats['vy']
        mag1 = np.sqrt(v1x**2 + v1y**2)
        mag2 = np.sqrt(v2x**2 + v2y**2)
        cos_sim = (v1x * v2x + v1y * v2y) / (mag1 * mag2 + 1e-5)
        cos_sim = cos_sim.clip(-1.0, 1.0).fillna(0.0)
        dir_change = (1.0 - cos_sim).where((mag1 >= 0.5) & (mag2 >= 0.5), 0.0)
        ball_stats['dir_change'] = dir_change

        return ball_stats

    def _compute_frame_features(self, df: pd.DataFrame, ball_stats: pd.DataFrame, all_frames: list) -> pd.DataFrame:
        """프레임별 선수 밀집도·골대 거리 계산."""
        frame_features = []
        for frame, frame_df in df.groupby('frame'):
            players = frame_df[frame_df['class_name'] == 'player']

            clustering_score = 0.0
            if len(players) >= 2:
                players_calc = players.copy()
                players_calc['y_bottom'] = players_calc['cy'] + players_calc['h'] / 2
                p_list = players_calc.to_dict('records')
                pair_distances = []
                for i in range(len(p_list)):
                    for j in range(i + 1, len(p_list)):
                        p1, p2 = p_list[i], p_list[j]
                        h_ratio = max(p1['h'], p2['h']) / (min(p1['h'], p2['h']) + 1e-5)
                        if h_ratio > 2.0:
                            pair_distances.append(10000.0)
                        else:
                            dx = p1['cx'] - p2['cx']
                            dy = (p1['y_bottom'] - p2['y_bottom']) * 3.0
                            pair_distances.append(np.sqrt(dx**2 + dy**2))
                avg_dist = np.mean(pair_distances) if pair_distances else 10000.0
                clustering_score = 1.0 / (avg_dist + 0.1)

            goals = frame_df[frame_df['class_name'] == 'goal']
            f_goalpost_visible = int(len(goals) > 0)
            f_dist_centroid_to_goal = np.nan
            f_dist_ball_to_goal = np.nan
            if not goals.empty:
                goal_center = np.array([goals.iloc[0]['cx'], goals.iloc[0]['cy']])
                active = frame_df[frame_df['class_name'].isin(['player', 'goalkeeper'])]
                if not active.empty:
                    centroid = np.array([active['cx'].mean(), active['cy'].mean()])
                    f_dist_centroid_to_goal = float(np.linalg.norm(centroid - goal_center))
                ball_row = frame_df[frame_df['class_name'] == 'ball']
                if not ball_row.empty:
                    ball_pos = np.array([ball_row.iloc[0]['cx'], ball_row.iloc[0]['cy']])
                    f_dist_ball_to_goal = float(np.linalg.norm(ball_pos - goal_center))

            frame_features.append({
                'frame': frame,
                'clustering': clustering_score,
                'audio': frame_df['f_audio'].max() if 'f_audio' in df.columns else 0.0,
                'f_goalpost_visible': f_goalpost_visible,
                'f_dist_centroid_to_goal': f_dist_centroid_to_goal,
                'f_dist_ball_to_goal': f_dist_ball_to_goal,
            })

        feat_df = pd.DataFrame(frame_features).set_index('frame').reindex(all_frames).fillna(0)

        feat_df['player_density'] = np.log1p(feat_df['clustering'])
        feat_df['player_density'] /= feat_df['player_density'].max() + 1e-5
        feat_df['f_audio'] = feat_df['audio'] / (feat_df['audio'].max() + 1e-5)

        norm_speed = ball_stats['speed'] / (ball_stats['speed'].max() + 1e-5)
        norm_accel = ball_stats['accel'] / (ball_stats['accel'].max() + 1e-5)
        feat_df['f_ball_speed'] = norm_speed
        feat_df['f_ball_accel'] = norm_accel
        feat_df['f_ball_dir_change'] = (
            ball_stats['dir_change'] / (ball_stats['dir_change'].max() + 1e-5)
        ).reindex(feat_df.index).fillna(0.0)

        visual_cols = ['f_ball_speed', 'f_ball_accel', 'player_density',
                       'f_dist_centroid_to_goal', 'f_dist_ball_to_goal']
        feat_df[visual_cols] = (feat_df[visual_cols]
                                .replace(0, np.nan)
                                .interpolate(method='linear', limit=15, limit_direction='both')
                                .fillna(0))
        return feat_df

    def _add_phase2_features(self, df: pd.DataFrame, feat_df: pd.DataFrame,
                              all_frames: list, ball_stats: pd.DataFrame) -> pd.DataFrame:
        """점유 전환(possession switch)·스프린트 수 계산 (ByteTrack 필요)."""
        player_types = ('player', 'goalkeeper')
        df_by_frame = {f: g for f, g in df.groupby('frame')}

        # 점유 전환
        poss_records = []
        for frame in all_frames:
            if frame not in ball_stats.index:
                poss_records.append((frame, -1)); continue
            bx, by = ball_stats.loc[frame, 'cx'], ball_stats.loc[frame, 'cy']
            if np.isnan(bx) or np.isnan(by):
                poss_records.append((frame, -1)); continue
            fdf = df_by_frame.get(frame)
            if fdf is None:
                poss_records.append((frame, -1)); continue
            pl = fdf[fdf['class_name'].isin(player_types)]
            pl = pl[pl['track_id'] != -1]
            if pl.empty:
                poss_records.append((frame, -1)); continue
            dists = np.sqrt((pl['cx'].values - bx)**2 + (pl['cy'].values - by)**2)
            poss_records.append((frame, int(pl['track_id'].iloc[int(np.argmin(dists))])))

        poss_df = pd.DataFrame(poss_records, columns=['frame', 'poss_id']).set_index('frame')
        prev_id = poss_df['poss_id'].shift(1)
        valid = (poss_df['poss_id'] != -1) & (prev_id != -1) & (poss_df['poss_id'] != prev_id)
        win = max(1, int(self.fps * 4 + 1))
        switch_rolling = valid.astype(int).rolling(window=win, center=True).sum().fillna(0)
        feat_df['f_possession_switches'] = (
            switch_rolling / (switch_rolling.max() + 1e-5)
        ).reindex(feat_df.index).fillna(0.0)

        # 스프린트 수 (절대 최소 속도 3px/frame 적용)
        players_df = df[df['class_name'] == 'player'].copy()
        if 'track_id' in players_df.columns and (players_df['track_id'] != -1).any():
            players_df = players_df[players_df['track_id'] != -1].sort_values(['track_id', 'frame'])
            players_df['p_speed'] = np.sqrt(
                players_df.groupby('track_id')['cx'].diff().pow(2) +
                players_df.groupby('track_id')['cy'].diff().pow(2)
            ).fillna(0.0)
            ABS_SPRINT_MIN = 3.0
            threshold = max(float(players_df['p_speed'].quantile(0.75)), ABS_SPRINT_MIN)
            sprint_count = (players_df[players_df['p_speed'] > threshold]
                            .groupby('frame').size()
                            .reindex(feat_df.index).fillna(0))
            feat_df['f_sprint_count'] = sprint_count / (sprint_count.max() + 1e-5)
        else:
            feat_df['f_sprint_count'] = 0.0

        return feat_df

    def _add_goal_relative_features(self, df: pd.DataFrame, feat_df: pd.DataFrame,
                                     all_frames: list, ball_stats: pd.DataFrame) -> pd.DataFrame:
        """Phase 3: 골대폭 정규화 기반 카메라 불변 피처."""
        player_types = ('player', 'goalkeeper')
        df_by_frame = {f: g for f, g in df.groupby('frame')}

        goal_rows = []
        for frame in all_frames:
            fdf = df_by_frame.get(frame)
            if fdf is None:
                continue
            goals_f = fdf[fdf['class_name'] == 'goal']
            if goals_f.empty:
                continue
            g = goals_f.loc[goals_f['w'].idxmax()]
            goal_rows.append({'frame': frame, 'g_cx': float(g['cx']), 'g_cy': float(g['cy']),
                               'g_w': float(g['w']), 'g_h': float(g['h'])})

        goal_ref_df = (pd.DataFrame(goal_rows).set_index('frame')
                       if goal_rows else pd.DataFrame(columns=['g_cx', 'g_cy', 'g_w', 'g_h']))
        goal_ref_df = goal_ref_df.reindex(all_frames)

        ball_cx_i = ball_stats['cx'].reindex(all_frames)
        ball_cy_i = ball_stats['cy'].reindex(all_frames)

        dist_px_goal = np.sqrt(
            (ball_cx_i - goal_ref_df['g_cx']).pow(2) +
            (ball_cy_i - goal_ref_df['g_cy']).pow(2)
        )
        feat_df['f_ball_to_goal_width_ratio'] = (
            dist_px_goal / (goal_ref_df['g_w'] + 1e-5)
        ).fillna(10.0).clip(upper=10.0)

        gw = goal_ref_df['g_w']
        feat_df['f_goal_bbox_width_norm'] = (gw / (gw.max() + 1e-5)).fillna(0.0)

        # 공 근처 선수 수 (골대폭 기반 반경)
        frame_to_idx = {f: i for i, f in enumerate(all_frames)}
        near_counts = np.zeros(len(all_frames), dtype=float)
        for frame in all_frames:
            if frame not in ball_stats.index:
                continue
            bx, by = ball_stats.loc[frame, 'cx'], ball_stats.loc[frame, 'cy']
            if pd.isna(bx) or pd.isna(by):
                continue
            fdf = df_by_frame.get(frame)
            if fdf is None:
                continue
            pl = fdf[fdf['class_name'].isin(player_types)]
            if pl.empty:
                continue
            pdists = np.sqrt((pl['cx'].values - bx)**2 + (pl['cy'].values - by)**2)
            gw_frame = goal_ref_df.loc[frame, 'g_w'] if frame in goal_ref_df.index else np.nan
            radius = (gw_frame * 1.5) if pd.notna(gw_frame) else 150.0
            near_counts[frame_to_idx[frame]] = int((pdists < radius).sum())

        near_series = pd.Series(near_counts, index=all_frames)
        feat_df['f_players_near_ball'] = near_series / (near_series.max() + 1e-5)

        return feat_df

    def _compute_rule_score(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """XH rule-based vision score 계산."""
        # 골대 가시성 ±2초 grace period 적용 후 centroid-goal 역거리
        # 골대 미탐지 프레임은 fillna(0)으로 채워졌으나 실제 거리=0은 불가 → 0은 미탐지로 처리
        dist_centroid = feat_df['f_dist_centroid_to_goal'].where(feat_df['f_dist_centroid_to_goal'] > 0)
        max_dist = dist_centroid.max() + 1e-5
        inv_dist_centroid = (1.0 - dist_centroid / max_dist).clip(lower=0).fillna(0.0)
        window_size = int(self.fps * 4 + 1)
        visible_in_window = feat_df['f_goalpost_visible'].rolling(window=window_size, center=True).max().fillna(0)
        # 골대 비가시 구간은 0.3배 soft penalty (하드 0 → 화면 밖 역습/중거리슛 FN 감소)
        visibility_factor = (visible_in_window > 0).astype(float) * 0.7 + 0.3
        inv_dist_centroid = inv_dist_centroid * visibility_factor
        feat_df['inv_dist_centroid_masked'] = inv_dist_centroid

        # Ball-in-box 보너스
        inv_dist_ball = 1.0 - feat_df['f_dist_ball_to_goal'] / (feat_df['f_dist_ball_to_goal'].max() + 1e-5)
        ball_goal_valid = feat_df['f_dist_ball_to_goal'] > 0  # 0 = 미탐지/보간 불가 → 제외
        cond1 = inv_dist_ball > 0.85        # 공이 골대 근접
        cond1_strong = (inv_dist_ball > 0.95) & ball_goal_valid  # 공이 골대 안/직전 — 골 상황
        cond2 = (feat_df['player_density'] > 0.2) & (inv_dist_centroid > 0.3)
        feat_df['bonus_score'] = 0.0
        feat_df.loc[cond1 & cond2, 'bonus_score'] = 0.5
        # 공이 골대 극근방이면 세리머니로 cond2 실패해도 보너스 부여
        feat_df.loc[cond1_strong, 'bonus_score'] = 0.7

        base_vision = (
            inv_dist_centroid * 0.35 +
            feat_df['player_density'] * 0.25 +
            feat_df['f_ball_accel'] * 0.20 +
            feat_df['f_ball_dir_change'] * 0.10 +
            feat_df['f_ball_speed'] * 0.05 +
            feat_df['f_players_near_ball'] * 0.05
        )
        vision_score = (base_vision + feat_df['bonus_score']).clip(upper=1.0)
        feat_df['xh_score'] = (vision_score * 0.9 + feat_df['f_audio'] * 0.1).clip(upper=1.0)

        top_feature_df = pd.DataFrame({
            'f_ball_speed': feat_df['f_ball_speed'],
            'f_ball_accel': feat_df['f_ball_accel'],
            'player_density': feat_df['player_density'],
            'f_players_near_ball': feat_df['f_players_near_ball'],
            'f_audio': feat_df['f_audio'],
            'inv_dist_centroid': inv_dist_centroid,
        })
        feat_df['top_feature'] = top_feature_df.idxmax(axis=1)

        return feat_df

    def _compute_ml_score(self, feat_df: pd.DataFrame) -> pd.DataFrame:
        """XGBoost 앙상블 스코어 계산."""
        feat_df['xgb_score'] = 0.0

        if not os.path.exists(AI_HIGHLIGHT_MODEL_PATH):
            feat_df['final_ensemble_score'] = feat_df['xh_score']
            return feat_df

        ai_features = (ML_FEATURES[:AI_MODEL_N_FEATURES]
                       if AI_MODEL_N_FEATURES and AI_MODEL_N_FEATURES < len(ML_FEATURES)
                       else ML_FEATURES)
        try:
            ai_data = []
            elite_features_data = []
            for i in range(len(feat_df)):
                start_idx = max(0, i - AI_EVENT_WINDOW_ROWS)
                end_idx = min(len(feat_df), i + AI_EVENT_WINDOW_ROWS + 1)
                window = feat_df.iloc[start_idx:end_idx]
                full_stats = {}
                for col in _ML_BASE_COLS:
                    if col in feat_df.columns:
                        series = window[col]
                        full_stats[f'{col}_anchor'] = float(feat_df.iloc[i][col])
                        full_stats[f'{col}_mean'] = float(series.mean())
                        full_stats[f'{col}_max'] = float(series.max())
                        full_stats[f'{col}_min'] = float(series.min())
                        full_stats[f'{col}_std'] = float(series.std(ddof=0))
                row_elite = {f: full_stats.get(f, 0.0) for f in ai_features}
                ai_data.append([row_elite[f] for f in ai_features])
                elite_features_data.append(row_elite)

            elite_df = pd.DataFrame(elite_features_data, index=feat_df.index)
            feat_df = pd.concat([feat_df, elite_df], axis=1)
            probs = AI_MODEL.predict_proba(np.array(ai_data))
            feat_df['xgb_score'] = probs[:, 1]
        except Exception as e:
            print(f"XGBoost 추론 중 오류 발생 (Rule-based로 대체): {e}")
            feat_df['xgb_score'] = feat_df['xh_score']

        if (feat_df['xgb_score'] == 0).all():
            feat_df['final_ensemble_score'] = feat_df['xh_score']
        else:
            feat_df['final_ensemble_score'] = feat_df['xh_score'] * 0.5 + feat_df['xgb_score'] * 0.5

        return feat_df

    def calculate_xh_score(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.sort_values('frame')
        all_frames = sorted(df['frame'].unique())

        if 'track_id' not in df.columns:
            df = df.copy()
            df['track_id'] = -1

        ball_stats = self._compute_ball_physics(df, all_frames)
        feat_df = self._compute_frame_features(df, ball_stats, all_frames)
        feat_df = self._add_phase2_features(df, feat_df, all_frames, ball_stats)
        feat_df = self._add_goal_relative_features(df, feat_df, all_frames, ball_stats)
        feat_df = self._compute_rule_score(feat_df)
        feat_df = self._compute_ml_score(feat_df)

        window_size = int(self.fps * 3)
        feat_df['smoothed_score'] = (feat_df['final_ensemble_score']
                                     .rolling(window=window_size, center=True)
                                     .mean().fillna(0))
        return feat_df.reset_index()


class HighlightExtractor:
    def __init__(self, fps: int = 30):
        self.fps = fps

    def extract_auto(
        self,
        xh_df: pd.DataFrame,
        count: int,
        min_interval_sec: int,
        exclude_intervals: list[tuple[float, float]] | None = None,
    ):
        stride = 10
        if len(xh_df) > 1:
            stride = xh_df['frame'].iloc[1] - xh_df['frame'].iloc[0]
            if stride <= 0:
                stride = 10

        peaks, _ = find_peaks(xh_df['smoothed_score'], distance=int(self.fps * 2 / stride))

        anchored_candidates = [
            {'frame': int(xh_df.loc[p_idx, 'frame']), 'score': xh_df.loc[p_idx, 'smoothed_score']}
            for p_idx in peaks
            if xh_df.loc[p_idx, 'smoothed_score'] >= SCORE_THRESHOLD
        ]

        if not anchored_candidates:
            candidates = xh_df.sort_values('smoothed_score', ascending=False).head(count)
            return sorted(candidates['frame'].tolist())

        candidates_df = pd.DataFrame(anchored_candidates).sort_values('score', ascending=False)
        highlight_frames = []
        for _, row in candidates_df.iterrows():
            f = int(row['frame'])
            if exclude_intervals and any(s <= f / self.fps <= e for s, e in exclude_intervals):
                continue
            if not any(abs(f - ef) < self.fps * min_interval_sec for ef in highlight_frames):
                highlight_frames.append(f)
            if len(highlight_frames) >= count:
                break

        return sorted(highlight_frames)


def get_audio_features(video_path, fps):
    print("오디오 데이터를 분석 중...")
    video = VideoFileClip(video_path)
    if video.audio is None:
        return None
    audio = video.audio
    sr = audio.fps
    audio_data = audio.to_soundarray()
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    samples_per_frame = sr / fps
    audio_features = []
    total_frames = int(video.duration * fps)
    for i in range(total_frames):
        start_idx, end_idx = int(i * samples_per_frame), int((i + 1) * samples_per_frame)
        frame_audio = audio_data[start_idx:end_idx]
        power = np.sqrt(np.mean(frame_audio**2)) if len(frame_audio) > 0 else 0.0
        audio_features.append({'frame': i, 'f_audio': float(power)})
    video.close()
    return pd.DataFrame(audio_features)


def detect_and_log_objects(video_path, model_path, stride: int = 7, use_tracker: bool = True):
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    fps, total_frames = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    all_detections = []

    tracker_ok = False
    if use_tracker:
        try:
            results = model.track(
                source=video_path, device=device, stream=True,
                vid_stride=stride, conf=CONF_THRESHOLD, imgsz=1280,
                verbose=False, persist=True, tracker="bytetrack.yaml",
            )
            tracker_ok = True
            desc = "객체 추적 중 (ByteTrack)"
        except Exception as e:
            print(f"[경고] ByteTrack 초기화 실패 → detect-only 로 폴백: {e}")

    if not tracker_ok:
        results = model.predict(
            source=video_path, device=device, stream=True,
            vid_stride=stride, conf=CONF_THRESHOLD, imgsz=1280, verbose=False,
        )
        desc = "객체 탐지 중"

    frame_count = 0
    for result in tqdm(results, total=total_frames // stride, desc=desc):
        current_frame = getattr(result, 'frame_idx', frame_count)
        if result.boxes is not None:
            ids_tensor = getattr(result.boxes, 'id', None)
            ids_list = ids_tensor.cpu().numpy().astype(int).tolist() if ids_tensor is not None else [-1] * len(result.boxes)
            for box, tid in zip(result.boxes, ids_list):
                coords = box.xyxy[0].cpu().numpy()
                all_detections.append({
                    'frame': int(current_frame),
                    'class_name': model.names[int(box.cls[0])],
                    'track_id': int(tid),
                    'cx': float((coords[0] + coords[2]) / 2),
                    'cy': float((coords[1] + coords[3]) / 2),
                    'w': float(coords[2] - coords[0]),
                    'h': float(coords[3] - coords[1]),
                })
        frame_count += stride
    return pd.DataFrame(all_detections), fps


def create_video_clips(video_path, frames, fps, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video = VideoFileClip(video_path)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_prefix = f"{base_name}_{time.strftime('%m%d_%H%M')}"

    created_paths = []
    for i, frame_idx in enumerate(tqdm(frames, desc="클립 생성 중")):
        t = frame_idx / fps
        start = max(0, t - CLIP_DURATION_BEFORE)
        end = min(video.duration, t + CLIP_DURATION_AFTER)
        try:
            clip = video.subclipped(start, end)
        except AttributeError:
            clip = video.subclip(start, end)
        output_path = os.path.join(output_dir, f"{video_prefix}_{i+1:02d}.mp4")
        clip.write_videofile(output_path, codec="libx264", audio_codec="aac",
                             temp_audiofile=f'temp-audio-{i}.m4a', remove_temp=True, logger=None)
        created_paths.append(output_path)
    video.close()
    return created_paths


def run_highlight_pipeline(
    video_path: str,
    model_path: str,
    output_dir: str,
    highlight_count: int = HIGHLIGHT_COUNT,
    exclude_intervals: list[tuple[float, float]] | None = None,
) -> HighlightRunResult:
    detections_df, fps = detect_and_log_objects(video_path, model_path)
    if detections_df.empty:
        return HighlightRunResult(False, "탐지된 객체가 없습니다.")

    if 'class_name' in detections_df.columns:
        print(f"골대 탐지 수: {(detections_df['class_name'] == 'goal').sum()}")

    audio_df = get_audio_features(video_path, fps)
    if audio_df is not None:
        detections_df = pd.merge(detections_df, audio_df, on='frame', how='left').fillna(0)
    else:
        detections_df['f_audio'] = 0.0

    calculator = XHScoreCalculator(XH_WEIGHTS, fps)
    xh_df = calculator.calculate_xh_score(detections_df)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    video_prefix = f"{base_name}_{time.strftime('%m%d_%H%M')}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    detections_df.to_csv(os.path.join(output_dir, f"{video_prefix}_detections.csv"), index=False)
    xh_df.to_csv(os.path.join(output_dir, f"{video_prefix}_xh_scores.csv"), index=False)

    extractor = HighlightExtractor(fps)
    highlight_frames = extractor.extract_auto(xh_df, highlight_count, MIN_SECONDS_BETWEEN_CLIPS, exclude_intervals)
    if not highlight_frames:
        return HighlightRunResult(False, "하이라이트 구간을 찾지 못했습니다.")

    clip_paths = create_video_clips(video_path, highlight_frames, fps, output_dir)

    mapping_df = pd.DataFrame({
        'clip_path': [os.path.basename(p) for p in clip_paths],
        'frame': highlight_frames,
    })
    mapping_df.to_csv(os.path.join(output_dir, f"{video_prefix}_clip_mapping.csv"), index=False)

    # XAI: Shapley 기여도 계산
    # XGB 모드: TreeExplainer로 실제 SHAP 값 (양수=기여, 음수=감점)
    # 룰 모드: weight × (클립값 - 영상평균) → "이 클립이 평균 대비 얼마나 특별한가"
    _RULE_CONTRIB = {
        'inv_dist_centroid_masked': 0.35,
        'player_density': 0.25,
        'f_ball_accel': 0.20,
        'f_ball_dir_change': 0.10,
        'f_ball_speed': 0.05,
        'f_players_near_ball': 0.05,
    }
    model_loaded = os.path.exists(AI_HIGHLIGHT_MODEL_PATH)
    ai_features_used = (ML_FEATURES[:AI_MODEL_N_FEATURES]
                        if AI_MODEL_N_FEATURES and AI_MODEL_N_FEATURES < len(ML_FEATURES)
                        else ML_FEATURES)

    # 룰 모드용 영상 전체 피처 평균 (baseline)
    rule_feature_means = {
        f: float(xh_df[f].mean()) if f in xh_df.columns else 0.0
        for f in _RULE_CONTRIB
    }

    clip_feature_stats: dict[int, dict[str, float]] = {}

    if model_loaded:
        try:
            import shap as _shap
            explainer = _shap.TreeExplainer(AI_MODEL)
            for frame in highlight_frames:
                row = xh_df[xh_df['frame'] == frame].head(1)
                if row.empty:
                    continue
                X_row = np.array([[float(row[f].iloc[0]) if f in xh_df.columns else 0.0
                                   for f in ai_features_used]])
                raw = explainer.shap_values(X_row)
                # binary classifier: list[class0, class1] 또는 단일 배열
                shap_vals = raw[1][0] if isinstance(raw, list) else raw[0]
                clip_feature_stats[frame] = dict(zip(ai_features_used, [float(v) for v in shap_vals]))
        except Exception as e:
            print(f"SHAP 계산 실패, 룰 기반으로 대체: {e}")
            model_loaded = False

    if not model_loaded:
        for frame in highlight_frames:
            row = xh_df[xh_df['frame'] == frame].head(1)
            if row.empty:
                continue
            # 영상 평균 대비 초과분 × 가중치 → 이 클립에서 특별히 튄 피처를 부각
            vals = {
                f: w * (float(row[f].iloc[0] if f in xh_df.columns else 0.0) - rule_feature_means[f])
                for f, w in _RULE_CONTRIB.items()
            }
            clip_feature_stats[frame] = vals

    # 최대 절댓값 기준으로 주요 피처 선정 (SHAP 음수 처리)
    clip_features = [
        max(clip_feature_stats[frame].items(), key=lambda x: abs(x[1]))[0]
        if frame in clip_feature_stats else 'unknown'
        for frame in highlight_frames
    ]

    clip_scores_by_frame: dict[int, float] = {}
    for frame in highlight_frames:
        row = xh_df[xh_df['frame'] == frame].head(1)
        if not row.empty:
            clip_scores_by_frame[frame] = float(row['smoothed_score'].iloc[0])

    status_msg = f"성공적으로 추출되었습니다. (하이라이트: {len(highlight_frames)}개)"
    if highlight_frames:
        f0 = highlight_frames[0]
        if f0 in clip_feature_stats:
            top_f = max(clip_feature_stats[f0].items(), key=lambda x: abs(x[1]))
            status_msg += f" | 주요 기여 피처: {top_f[0]} ({top_f[1]*100:.1f}%)"

    return HighlightRunResult(True, status_msg, fps, highlight_frames,
                              output_dir, clip_paths, clip_features, clip_feature_stats,
                              clip_scores_by_frame)


def main():
    print("--- XH 기반 축구 하이라이트 시스템 ---")
    result = run_highlight_pipeline(VIDEO_PATH, MODEL_PATH, OUTPUT_DIR)
    print(result.message)


if __name__ == "__main__":
    main()

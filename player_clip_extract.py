"""축구 선수 개인 클립 추출 — 탐지/추적 기반 (농구 로그 컷 대체).

흐름: 영상에서 player+ball 을 탐지·추적(ByteTrack track_id) → 사용자가 UI 에서
지정한 선수 track_id 들이 '공에 관여'(공 보유자=공에 가장 가깝고 반경 이내)한
구간만 잘라낸다. XH 점수/위협도와 무관하게 "그 선수가 공에 관여한 장면"만 추출.

추적이 끊기면(가림·화면전환) UI 에서 다시 클릭해 같은 선수 세션에 track_id 를
이어붙이므로, 여기서는 넘어온 track_id 집합 전체를 한 선수로 취급한다.

끊김·재클릭을 줄이려 2단 방어:
  1) BoT-SORT+ReID(botsort_player.yaml): 외형 임베딩으로 가림·교차 후 같은 선수 재연결
     → 모션/IoU만 보는 ByteTrack 의 교차 시 ID 스위치를 근본 완화.
  2) stitch_player_tracks: ReID 가 놓친 짧은 가림 구간을, 끊긴 트랙의 마지막 속도로
     위치를 외삽해 직후 재등장 트랙과 이어붙여 track_id 정규화(탐지 직후 1회).
"""
from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from extract import detect_and_log_objects, MODEL_PATH

# 트랙 수명↑(끊김·재클릭↓)용 커스텀 추적 설정. 선호도순: BoT-SORT+ReID(가림·교차
# 후 외형으로 재연결) → ByteTrack(모션) → ultralytics 기본. 존재하는 첫 파일 사용.
_HERE = os.path.dirname(os.path.abspath(__file__))
def _first_existing(*names: str) -> str:
    for n in names:
        p = os.path.join(_HERE, n)
        if os.path.exists(p):
            return p
    return "bytetrack.yaml"
PLAYER_TRACKER = _first_existing("botsort_player.yaml", "bytetrack_player.yaml")

# 공 관여 판정: 공-선수 거리가 (선수 bbox 높이 × 배수) 이내면 관여 (원근 보정).
INVOLVE_HEIGHT_FACTOR = 2.5
INVOLVE_MIN_PX = 60.0          # 선수가 작게 잡혀도 최소 반경
SEGMENT_MERGE_GAP_SEC = 2.0   # 관여 시점 간 이 간격 이내면 한 구간으로 병합
CLIP_PAD_BEFORE = 3.0         # 관여 구간 앞 패딩(초)
CLIP_PAD_AFTER = 3.0          # 관여 구간 뒤 패딩(초)
# 하이라이트 추출(stride=7)과 달리 개인 추적은 트랙 연속성이 생명 → 작은 stride 사용.
# stride=7/60fps 에선 ByteTrack ID가 ~2초마다 끊겨 재클릭이 잦았음(검증). 3으로 완화.
PLAYER_STRIDE = 3
DETECTION_FILE = "player_detections.csv"

# --- 트랙 스티칭(사후 ID 병합) 파라미터 ---
# 트래커가 가림·교차로 track_id 를 끊었다 새로 만들 때, 끊긴 트랙의 마지막 속도로
# 위치를 외삽해 곧 재등장한 새 트랙과 이어붙인다(같은 선수로 간주). ReID 트래커가
# 잡아주는 것의 안전망 — ReID 가 놓친 짧은 가림 구간을 모션 예측으로 메운다.
STITCH_MAX_GAP_SEC = 1.5        # 끊긴 뒤 이 시간 내 재등장한 트랙만 이어붙임
STITCH_DIST_HEIGHT_FACTOR = 3.0 # 예측 위치 허용 오차(선수 키 배수, 원근 보정)
STITCH_VEL_SAMPLES = 5          # 끝점 속도 추정에 쓸 마지막 샘플 수


@dataclass
class PlayerDetectResult:
    success: bool
    message: str
    fps: float = 30.0
    detection_file: str = ""
    n_frames: int = 0
    n_player_tracks: int = 0


def _video_duration(video_path: str) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def _cut_clip(video_path: str, start: float, duration: float, out_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video_path,
        "-t", f"{duration:.3f}", "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "make_zero", out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


def run_player_detection(video_path: str, output_dir: str,
                         model_path: str | None = None) -> PlayerDetectResult:
    """영상 탐지·추적 → detections CSV 저장. (무겁다: 풀경기는 수 분)"""
    det_df, fps = detect_and_log_objects(video_path, model_path or MODEL_PATH,
                                         stride=PLAYER_STRIDE, tracker=PLAYER_TRACKER)
    if det_df is None or det_df.empty:
        return PlayerDetectResult(False, "탐지된 객체가 없습니다.")
    # 가림·교차로 끊긴 트랙을 모션 외삽으로 이어붙여 track_id 정규화(재클릭↓)
    det_df, n_stitched = stitch_player_tracks(det_df, fps)
    if n_stitched:
        print(f"트랙 스티칭: 끊긴 트랙 {n_stitched}쌍 병합")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    det_df.to_csv(out / DETECTION_FILE, index=False)
    players = det_df[det_df["class_name"] == "player"]
    n_tracks = int(players[players["track_id"] != -1]["track_id"].nunique())
    return PlayerDetectResult(
        success=True,
        message=f"탐지 완료 (player track {n_tracks}개, 공 탐지 {int((det_df['class_name']=='ball').sum())}프레임)",
        fps=float(fps),
        detection_file=DETECTION_FILE,
        n_frames=int(det_df["frame"].nunique()),
        n_player_tracks=n_tracks,
    )


def _endpoint_velocity(g: pd.DataFrame, n: int) -> tuple[float, float]:
    """트랙 끝부분 마지막 n 샘플로 프레임당 (vx, vy) px 추정. 샘플<2면 (0,0)."""
    tail = g.tail(n)
    if len(tail) < 2:
        return 0.0, 0.0
    df = float(tail["frame"].iloc[-1] - tail["frame"].iloc[0])
    if df <= 0:
        return 0.0, 0.0
    vx = float(tail["cx"].iloc[-1] - tail["cx"].iloc[0]) / df
    vy = float(tail["cy"].iloc[-1] - tail["cy"].iloc[0]) / df
    return vx, vy


def stitch_player_tracks(det_df: pd.DataFrame, fps: float) -> tuple[pd.DataFrame, int]:
    """가림·교차로 끊긴 player track_id 를 모션 외삽으로 이어붙여 정규화.

    끊긴 트랙 A(마지막 위치+속도)의 위치를 공백 길이만큼 외삽해, 직후 등장한 트랙 B의
    시작 위치가 허용 오차 내면 같은 선수로 보고 병합(1:1 체이닝 → 동시 존재 트랙 오병합 방지).
    player 행의 track_id 를 대표 id(체인 내 최소 원본 id)로 덮어쓴 df 와 병합 횟수 반환.
    """
    if det_df.empty or fps <= 0:
        return det_df, 0
    players = det_df[(det_df["class_name"] == "player") & (det_df["track_id"] != -1)]
    if players.empty:
        return det_df, 0

    max_gap = STITCH_MAX_GAP_SEC * fps
    info: dict[int, dict] = {}
    for tid, g in players.groupby("track_id"):
        g = g.sort_values("frame")
        vx, vy = _endpoint_velocity(g, STITCH_VEL_SAMPLES)
        info[int(tid)] = {
            "start_f": int(g["frame"].iloc[0]), "end_f": int(g["frame"].iloc[-1]),
            "start": (float(g["cx"].iloc[0]), float(g["cy"].iloc[0])),
            "end": (float(g["cx"].iloc[-1]), float(g["cy"].iloc[-1])),
            "vel": (vx, vy), "h": float(g["h"].median()),
        }

    # 후보 (A→B) 쌍: A 종료 후 max_gap 내 시작한 B, A 예측위치와 B 시작위치 거리 < 임계
    cands: list[tuple[float, int, int]] = []
    ids = list(info.keys())
    for a in ids:
        ia = info[a]
        for b in ids:
            if a == b:
                continue
            ib = info[b]
            gap = ib["start_f"] - ia["end_f"]
            if gap <= 0 or gap > max_gap:
                continue
            px = ia["end"][0] + ia["vel"][0] * gap
            py = ia["end"][1] + ia["vel"][1] * gap
            dist = ((ib["start"][0] - px) ** 2 + (ib["start"][1] - py) ** 2) ** 0.5
            # 허용 오차: 선수 키 배수, 공백이 길수록 완화
            tol = max(INVOLVE_MIN_PX, ia["h"] * STITCH_DIST_HEIGHT_FACTOR) * (1 + gap / max_gap)
            if dist <= tol:
                cands.append((dist, a, b))

    # 거리 가까운 순 greedy 1:1 체이닝 (A 후속·B 선행 각 1개) → union-find
    parent = {t: t for t in ids}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    has_succ: set[int] = set()
    has_pred: set[int] = set()
    n_merged = 0
    for _, a, b in sorted(cands):
        if a in has_succ or b in has_pred:
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        parent[max(ra, rb)] = min(ra, rb)  # 대표 = 최소 원본 id
        has_succ.add(a); has_pred.add(b); n_merged += 1

    if n_merged == 0:
        return det_df, 0
    remap = {t: find(t) for t in ids}
    det_df = det_df.copy()
    mask = (det_df["class_name"] == "player") & (det_df["track_id"] != -1)
    det_df.loc[mask, "track_id"] = det_df.loc[mask, "track_id"].map(
        lambda t: remap.get(int(t), int(t)))
    return det_df, n_merged


def _ball_possessor_frames(det_df: pd.DataFrame) -> list[tuple[int, int, float, float, float]]:
    """프레임별 '공 보유자'(공에 최근접 & 반경 이내) 목록 → (frame, track_id, cx, cy, h).

    공 관여 판정의 단일 소스. _involvement_frames(특정 선수 필터)와
    compute_possession_events(전체 이벤트 열거)가 모두 이 결과를 쓴다. frame 오름차순.
    """
    ball = (det_df[det_df["class_name"] == "ball"]
            .drop_duplicates("frame").set_index("frame")[["cx", "cy"]])
    players = det_df[(det_df["class_name"] == "player") & (det_df["track_id"] != -1)]
    out: list[tuple[int, int, float, float, float]] = []
    for frame, pl in players.groupby("frame"):
        if frame not in ball.index:
            continue
        bx, by = ball.loc[frame, "cx"], ball.loc[frame, "cy"]
        d = np.sqrt((pl["cx"] - bx) ** 2 + (pl["cy"] - by) ** 2)
        idx = d.idxmin()
        radius = max(INVOLVE_MIN_PX, float(pl.loc[idx, "h"]) * INVOLVE_HEIGHT_FACTOR)
        if float(d.loc[idx]) <= radius:
            out.append((int(frame), int(pl.loc[idx, "track_id"]),
                        float(pl.loc[idx, "cx"]), float(pl.loc[idx, "cy"]),
                        float(pl.loc[idx, "h"])))
    return out


def _involvement_frames(det_df: pd.DataFrame,
                        track_windows: list[tuple[int, float, float]]) -> list[int]:
    """지정 선수가 '공 보유자'(공에 최근접 & 반경 이내)인 프레임 목록.

    track_windows: (track_id, from_frame, to_frame) 목록. 같은 track_id 라도 그
    프레임 구간에 속할 때만 해당 선수로 인정한다. ByteTrack 은 가림·화면전환 후 끊긴
    id 를 다른 선수에게 재사용하므로(검증: 한 id 가 영상 내 여러 분리 구간에 등장),
    구간을 묶지 않으면 같은 id 의 다른 선수 장면까지 섞인다. 한 선수가 여러 id 를
    거치면 각 id 의 해당 구간을 windows 로 여러 개 넘기면 모두 누적된다.
    """
    from collections import defaultdict
    windows: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for tid, f0, f1 in track_windows:
        windows[int(tid)].append((float(f0), float(f1)))
    if not windows:
        return []
    return sorted(
        frame for frame, tid, *_ in _ball_possessor_frames(det_df)
        if tid in windows and any(f0 <= frame <= f1 for f0, f1 in windows[tid])
    )


def compute_possession_events(det_df: pd.DataFrame, fps: float) -> list[dict]:
    """영상 전체의 '공 보유 이벤트' 열거 — 선수 지정 없이 클립 후보 전부를 뽑는다.

    같은 track_id 가 연속(≤SEGMENT_MERGE_GAP_SEC 공백)으로 공을 보유한 구간 = 1 이벤트.
    옵션 A(이벤트 리뷰) 의 후보 소스: 사용자는 각 이벤트가 타깃 선수인지만 판정한다.
    각 dict: id/track_id/start/end(초)/start_frame/end_frame/mid_frame/mid_cx/mid_cy/n_touch.
    """
    poss = _ball_possessor_frames(det_df)
    if not poss or fps <= 0:
        return []
    gap_f = SEGMENT_MERGE_GAP_SEC * fps
    runs: list[dict] = []
    cur: dict | None = None
    for frame, tid, cx, cy, _h in poss:
        if cur and tid == cur["track_id"] and frame - cur["end_frame"] <= gap_f:
            cur["end_frame"] = frame
            cur["pts"].append((frame, cx, cy))
        else:
            if cur:
                runs.append(cur)
            cur = {"track_id": tid, "start_frame": frame, "end_frame": frame,
                   "pts": [(frame, cx, cy)]}
    if cur:
        runs.append(cur)

    events: list[dict] = []
    for i, r in enumerate(runs):
        mf, mcx, mcy = r["pts"][len(r["pts"]) // 2]  # 대표(중간) 터치 프레임
        events.append({
            "id": i, "track_id": r["track_id"],
            "start": round(r["start_frame"] / fps, 1),
            "end": round(r["end_frame"] / fps, 1),
            "start_frame": r["start_frame"], "end_frame": r["end_frame"],
            "mid_frame": mf, "mid_cx": round(mcx, 1), "mid_cy": round(mcy, 1),
            "n_touch": len(r["pts"]),
        })
    return events


def compute_player_segments(det_df: pd.DataFrame, fps: float,
                            track_ids: list[int] | None,
                            pad_before: float = CLIP_PAD_BEFORE,
                            pad_after: float = CLIP_PAD_AFTER,
                            exclude_intervals: list[tuple[float, float]] | None = None,
                            track_windows: list[tuple[int, float, float]] | None = None,
                            video_duration: float | None = None,
                            extra_involve_secs: list[float] | None = None) -> list[dict]:
    """관여 프레임 → 제외 적용 → 구간 병합 → 패딩 → 클립 메타 목록 반환(컷 없음).

    extract_player_clips 와 동일한 구간 산정 로직 — 추출 전 '예상 클립 수' 미리보기에
    재사용한다(ffmpeg 컷만 생략). 각 dict: start/end(패딩 적용 클립 범위), involve_start/end.
    track_windows/track_ids 의미는 extract_player_clips 와 동일.

    extra_involve_secs: 사용자가 직접 찍은(공 미탐지로 자동 이벤트가 없는) 관여 시각(초).
    트래킹/공탐지와 무관한 '눈으로 보증한 터치' — 관여 시각 풀에 합쳐 같은 병합/패딩을 거친다.
    """
    if track_windows:
        windows = [(int(t), float(a), float(b)) for t, a, b in track_windows]
    else:
        fmax = float(det_df["frame"].max()) if not det_df.empty else 0.0
        windows = [(int(t), 0.0, fmax) for t in (track_ids or [])]
    frames = _involvement_frames(det_df, windows)

    secs = sorted({f / fps for f in frames})
    if exclude_intervals:
        ex = [(float(a), float(b)) for a, b in exclude_intervals if float(b) > float(a)]
        if ex:
            secs = [s for s in secs if not any(a <= s <= b for a, b in ex)]
    if extra_involve_secs:  # 사용자 보증 마크(제외 영향 안 받음)
        secs = sorted(set(secs) | {round(float(s), 3) for s in extra_involve_secs})
    if not secs:
        return []
    merged: list[tuple[float, float]] = []
    s = e = secs[0]
    for t in secs[1:]:
        if t - e <= SEGMENT_MERGE_GAP_SEC:
            e = t
        else:
            merged.append((s, e)); s = e = t
    merged.append((s, e))

    segments: list[dict] = []
    for seg_s, seg_e in merged:
        start = max(0.0, seg_s - pad_before)
        end = seg_e + pad_after
        if video_duration:
            end = min(end, video_duration)
        if end - start <= 0:
            continue
        segments.append({
            "start": round(start, 1), "end": round(end, 1),
            "involve_start": round(seg_s, 1), "involve_end": round(seg_e, 1),
        })
    return segments


def extract_player_clips(video_path: str, det_df: pd.DataFrame, fps: float,
                         track_ids: list[int] | None, output_dir: str,
                         pad_before: float = CLIP_PAD_BEFORE,
                         pad_after: float = CLIP_PAD_AFTER,
                         exclude_intervals: list[tuple[float, float]] | None = None,
                         track_windows: list[tuple[int, float, float]] | None = None,
                         extra_involve_secs: list[float] | None = None
                         ) -> tuple[list[str], list[dict]]:
    """관여 프레임 → 구간 병합 → 패딩 → 클립 컷. (clip_paths, segment_meta) 반환.

    track_windows: (track_id, from_frame, to_frame) 목록 — 시간 구간에 묶인 선수 지정.
    같은 id 가 화면전환 후 다른 선수에게 재사용돼도 그 구간만 인정해 오염을 막는다.
    track_windows 가 없으면 track_ids 를 영상 전체 구간으로 간주(구버전 호환).

    exclude_intervals: 신원 검증에서 "잘못 따라간(드리프트)" 것으로 표시한 (시작,끝)초 구간.
    해당 시각의 관여 프레임은 버린다 → 엉뚱한 선수 장면이 납품물에 섞이는 것 방지.
    extra_involve_secs: 공 미탐지로 자동 이벤트가 없는 구간을 사용자가 직접 찍은 관여 시각.
    """
    duration = _video_duration(video_path)
    segments = compute_player_segments(
        det_df, fps, track_ids, pad_before, pad_after,
        exclude_intervals, track_windows, video_duration=duration,
        extra_involve_secs=extra_involve_secs)
    if not segments:
        return [], []

    out = Path(output_dir) / "clips"
    out.mkdir(parents=True, exist_ok=True)
    prefix = f"player_{time.strftime('%m%d_%H%M')}"
    clip_paths: list[str] = []
    seg_meta: list[dict] = []
    for i, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        name = f"{prefix}_{int(seg['involve_start']):05d}s_{i:03d}.mp4"
        if _cut_clip(video_path, start, end - start, str(out / name)):
            clip_paths.append(str(out / name))
            seg_meta.append({"clip": name, **seg})
    return clip_paths, seg_meta

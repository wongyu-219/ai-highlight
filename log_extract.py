"""Log-based highlight extraction pipeline.

타임라인 로그(elapsedSeconds + halfType + eventLog)를 파싱해
이벤트 기준 -10s ~ +5s 클립을 ffmpeg으로 추출하고,
부족분을 AI(YOLO)로 보충해 target_count개를 맞춘다.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

CLIP_BEFORE = 45   # 이벤트 기준 앞으로 자를 초
CLIP_AFTER  = 25   # 이벤트 기준 뒤로 자를 초

# AI 클립 크기 (extract.py CLIP_DURATION_BEFORE/AFTER 와 동일 값)
_AI_CLIP_BEFORE = 15
_AI_CLIP_AFTER  = 10

VALID_EVENTS = {"GOAL", "HIGHLIGHT", "VALID_SHOT"}
_MIN_GAP = 20  # 이벤트 병합 최소 간격 (초). 20s 이내 이벤트는 병합(GOAL 우선)

# 로그 클립 스코어 (이벤트 우선순위 기반, AI XH score 범위보다 높게)
_EVENT_SCORES = {"GOAL": 1.0, "HIGHLIGHT": 0.9, "VALID_SHOT": 0.8}


@dataclass
class LogEvent:
    video_sec: float
    event_type: str
    half: str
    elapsed_sec: int


@dataclass
class LogExtractResult:
    success: bool
    message: str
    clip_paths: list[str] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    fps: float = 30.0
    clip_scores: dict[str, float] = field(default_factory=dict)          # {clip_name: score}
    clip_features: dict[str, str] = field(default_factory=dict)          # {clip_name: top_feature}
    clip_feature_stats: dict[str, dict] = field(default_factory=dict)    # {clip_name: {feature: weight}}
    highlight_frames: list[int] = field(default_factory=list)            # AI 앵커 프레임


# ──────────────────────────────────────────────
# 파싱
# ──────────────────────────────────────────────

def parse_log(
    log_data: list[dict[str, Any]],
    second_half_start_sec: float,
    event_filter: set[str] | None = None,
) -> list[LogEvent]:
    """API 응답 data 배열을 받아 이벤트 리스트 반환.

    정렬: FIRST_HALF elapsedSeconds 오름차순 → SECOND_HALF elapsedSeconds 오름차순
    """
    if event_filter is None:
        event_filter = VALID_EVENTS

    events: list[LogEvent] = []
    for item in log_data:
        event_type = str(item.get("eventLog", "")).upper()
        if event_type not in event_filter:
            continue
        # API 필드명은 elapsedSeconds (구버전 elapsedTime 폴백)
        elapsed_sec = int(item.get("elapsedSeconds", item.get("elapsedTime", 0)))
        half = str(item.get("halfType", "FIRST_HALF")).upper()

        if half == "FIRST_HALF":
            video_sec = float(elapsed_sec)
        else:
            video_sec = second_half_start_sec + elapsed_sec

        events.append(LogEvent(
            video_sec=video_sec,
            event_type=event_type,
            half=half,
            elapsed_sec=elapsed_sec,
        ))

    # FIRST_HALF 시간순 → SECOND_HALF 시간순
    half_order = {"FIRST_HALF": 0, "SECOND_HALF": 1}
    events.sort(key=lambda e: (half_order.get(e.half, 2), e.elapsed_sec))
    return events


def _dedup(events: list[LogEvent]) -> list[LogEvent]:
    """시간순 정렬 후 _MIN_GAP 이내 이벤트는 병합 (GOAL > SHOT 우선)."""
    if not events:
        return []
    sorted_ev = sorted(events, key=lambda e: e.video_sec)
    result = [sorted_ev[0]]
    for ev in sorted_ev[1:]:
        prev = result[-1]
        if ev.video_sec - prev.video_sec < _MIN_GAP:
            if ev.event_type == "GOAL" and prev.event_type != "GOAL":
                result[-1] = ev
        else:
            result.append(ev)
    return result


# ──────────────────────────────────────────────
# ffmpeg 유틸
# ──────────────────────────────────────────────

def _get_video_duration(video_path: str) -> float | None:
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        return float(r.stdout.strip())
    except Exception:
        return None


def _get_video_fps(video_path: str) -> float | None:
    """ffprobe 의 r_frame_rate(예: '60000/1001')를 실수 fps 로 반환. 실패 시 None.

    AI 클립 anchor 라벨(frame/fps)을 정확히 계산하려면 실제 fps 가 필요하다.
    (예: 59.94fps 영상을 30fps 로 잘못 나누면 라벨 시각이 2배가 됨)
    """
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=r_frame_rate",
             "-of", "default=noprint_wrappers=1:nokey=1", video_path],
            capture_output=True, text=True, check=True,
        )
        raw = r.stdout.strip()
        if "/" in raw:
            num, den = raw.split("/")
            den_f = float(den)
            return float(num) / den_f if den_f else None
        return float(raw)
    except Exception:
        return None


def _cut_clip(video_path: str, start: float, duration: float, out_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{start:.3f}",
        "-i", video_path,
        "-t", f"{duration:.3f}",
        "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError:
        return False


# ──────────────────────────────────────────────
# 메인 파이프라인
# ──────────────────────────────────────────────

def run_log_pipeline(
    video_path: str,
    log_data: list[dict[str, Any]],
    second_half_start_sec: float,
    output_dir: str,
    target_count: int = 40,
    model_path: str | None = None,
) -> LogExtractResult:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 파싱 + 중복 제거
    events = _dedup(parse_log(log_data, second_half_start_sec))
    if not events:
        return LogExtractResult(
            success=False,
            message="로그에서 유효한 이벤트(GOAL/SHOT)를 찾지 못했습니다.",
        )

    video_duration = _get_video_duration(video_path)
    prefix = f"log_{time.strftime('%m%d_%H%M')}"
    clip_paths: list[str] = []
    events_meta: list[dict] = []

    # 2. ffmpeg 클립 추출
    print(f"[로그] 이벤트 {len(events)}개 → 클립 추출 시작")
    for i, ev in enumerate(tqdm(events, desc="로그 클립 추출 중")):
        start = max(0.0, ev.video_sec - CLIP_BEFORE)
        end = ev.video_sec + CLIP_AFTER
        if video_duration:
            end = min(end, video_duration)
        duration = end - start
        if duration <= 0:
            continue

        clip_name = (
            f"{prefix}_{ev.half.lower()}_{ev.elapsed_sec:04d}s"
            f"_{ev.event_type.lower()}_{i:03d}.mp4"
        )
        clip_path = str(out_dir / clip_name)

        if _cut_clip(video_path, start, duration, clip_path):
            clip_paths.append(clip_path)
            events_meta.append({
                "clip": clip_name,
                "event_type": ev.event_type,
                "half": ev.half,
                "elapsed_sec": ev.elapsed_sec,
                "video_sec": ev.video_sec,
                "source": "log",
            })

    log_count = len(clip_paths)
    remaining = target_count - log_count
    ai_count = 0
    _ai_result = None

    # 3. AI 보충 (로그 클립 구간은 extract_auto 내부에서 pre-filter)
    if remaining > 0 and model_path:
        exclude_intervals = [
            (
                max(0.0, ev.video_sec - CLIP_BEFORE - _AI_CLIP_BEFORE),
                ev.video_sec + CLIP_AFTER + _AI_CLIP_AFTER,
            )
            for ev in events
        ]
        try:
            from extract import run_highlight_pipeline  # noqa: PLC0415
            ai_result = run_highlight_pipeline(
                video_path=video_path,
                model_path=model_path,
                output_dir=output_dir,
                highlight_count=remaining,
                exclude_intervals=exclude_intervals,
            )
            if ai_result.success and ai_result.clip_paths:
                for p in ai_result.clip_paths:
                    clip_paths.append(p)
                    events_meta.append({"clip": Path(p).name, "source": "ai"})
                ai_count = len(ai_result.clip_paths)
                _ai_result = ai_result
        except Exception as exc:
            print(f"[log_extract] AI 보충 실패: {exc}")

    # 4. 스코어 / 피처 데이터 구축
    clip_scores: dict[str, float] = {}
    clip_features_map: dict[str, str] = {}
    clip_feature_stats_map: dict[str, dict] = {}
    highlight_frames: list[int] = []

    for meta in events_meta:
        if meta.get("source") == "log":
            name = meta["clip"]
            clip_scores[name] = _EVENT_SCORES.get(meta.get("event_type", ""), 0.5)
            clip_features_map[name] = meta.get("event_type", "").lower() or "log"

    if _ai_result:
        ai_frames = _ai_result.highlight_frames or []
        ai_feats = _ai_result.clip_features or []
        ai_scores_map = _ai_result.clip_scores or {}
        ai_feat_stats = _ai_result.clip_feature_stats or {}

        for i, p in enumerate(_ai_result.clip_paths or []):
            name = Path(p).name
            frame = ai_frames[i] if i < len(ai_frames) else None
            if frame is not None:
                clip_scores[name] = ai_scores_map.get(frame, 0.0)
                highlight_frames.append(frame)
                if frame in ai_feat_stats:
                    clip_feature_stats_map[name] = ai_feat_stats[frame]
            if i < len(ai_feats):
                clip_features_map[name] = ai_feats[i]

    # 실제 fps: AI 보충 결과의 fps(탐지 시 cv2 로 읽은 값) 우선, 없으면 ffprobe 로 직접 측정.
    # 서버가 AI 클립 anchor 라벨을 frame/fps 로 계산하므로 실제 fps 가 정확해야 함.
    real_fps = (_ai_result.fps if _ai_result and _ai_result.fps else None) or _get_video_fps(video_path) or 30.0

    return LogExtractResult(
        success=True,
        message=f"로그 {log_count}개 + AI {ai_count}개 클립 추출 완료",
        clip_paths=clip_paths,
        events=events_meta,
        fps=float(real_fps),
        clip_scores=clip_scores,
        clip_features=clip_features_map,
        clip_feature_stats=clip_feature_stats_map,
        highlight_frames=highlight_frames,
    )

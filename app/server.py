from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from moviepy import VideoFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips, vfx, afx
from pydantic import BaseModel

from extract import run_highlight_pipeline, USE_XGB
from log_extract import (
    run_log_pipeline,
    CLIP_BEFORE as LOG_CLIP_BEFORE,
    CLIP_AFTER as LOG_CLIP_AFTER,
)
from player_clip_extract import (
    run_player_detection,
    extract_player_clips,
    compute_player_segments,
    compute_possession_events,
    DETECTION_FILE as PLAYER_DETECTION_FILE,
    CLIP_PAD_BEFORE as PLAYER_PAD_BEFORE,
    CLIP_PAD_AFTER as PLAYER_PAD_AFTER,
)

BASE_DIR = Path(__file__).resolve().parents[1]
RUNS_DIR = BASE_DIR / "runs"
UPLOADS_DIR = BASE_DIR / "uploads"
EXPORTS_DIR = BASE_DIR / "exports"
HIGHLIGHTS_DIR = BASE_DIR / "highlights"
MODEL_PATH = BASE_DIR / "best-8.pt"
EVENT_MARKS_PATH = BASE_DIR / "event_marks.json"
BGM_DIR = BASE_DIR / "app" / "static" / "bgm"
BGM_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_HIGHLIGHT_COUNT = 40
DEFAULT_MIN_INTERVAL = 15

# 합치기(export) 출력 해상도 고정. 해상도가 섞인 클립(예: 4K 외부 클립 + 1080p 하이라이트)을
# 그대로 concatenate(method="compose") 하면 캔버스가 가장 큰 해상도로 잡혀 작은 클립이
# 검은 캔버스 가운데에 축소되어 박힘. 모든 클립을 이 해상도로 통일(레터박스 유지)해 해결.
EXPORT_TARGET_SIZE = (1920, 1080)

# 클립 생성 시 앵커 기준 앞뒤 여유 (extract.py와 일치)
CLIP_DURATION_BEFORE = 15
CLIP_DURATION_AFTER = 10

# Option B: 표준 이벤트 타입 (축구 하이라이트 기준 10종)
EVENT_TYPES = [
    "goal",       # 골
    "shot",       # 슈팅
    "save",       # 선방
    "setpiece",   # 세트피스
    "threat",     # 위협 전개
    "defense",    # 수비
]
EVENT_TYPE_SET = set(EVENT_TYPES) | {"unknown", "none"}

RUNS_DIR.mkdir(exist_ok=True, parents=True)
UPLOADS_DIR.mkdir(exist_ok=True, parents=True)
EXPORTS_DIR.mkdir(exist_ok=True, parents=True)
(HIGHLIGHTS_DIR / "Good").mkdir(exist_ok=True, parents=True)
(HIGHLIGHTS_DIR / "Bad").mkdir(exist_ok=True, parents=True)

app = FastAPI(title="AI Highlight Server", version="0.1.0")
app.mount("/runs", StaticFiles(directory=RUNS_DIR), name="runs")
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")


def _job_dir(job_id: str) -> Path:
    return RUNS_DIR / job_id


def _metadata_path(job_id: str) -> Path:
    return _job_dir(job_id) / "metadata.json"


def _load_metadata(job_id: str) -> dict[str, Any]:
    path = _metadata_path(job_id)
    if not path.exists():
        return {
            "job_id": job_id,
            "status": "unknown",
            "message": "작업 정보를 찾지 못했습니다.",
            "clips": [],
            "selected": {},
            "created_at": None,
        }
    return json.loads(path.read_text(encoding="utf-8"))


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def _save_metadata(job_id: str, data: dict[str, Any]) -> None:
    path = _metadata_path(job_id)
    payload = json.dumps(data, ensure_ascii=False, indent=2, cls=_NumpyEncoder)
    # Write to a temp file in the same dir, then atomically replace. Plain
    # write_text isn't atomic, so concurrent saves (FastAPI threadpool) could
    # interleave and leave a corrupted half-written file.
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".metadata.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _require_job(job_id: str) -> Path:
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다.")
    return job_dir


def _safe_clip_path(job_dir: Path, clip_name: str) -> Path:
    clip_path = (job_dir / "clips" / clip_name).resolve()
    if not str(clip_path).startswith(str((job_dir / "clips").resolve())):
        raise HTTPException(status_code=400, detail="잘못된 파일 경로입니다.")
    return clip_path


def _locate_clip(job_id: str, clip_name: str) -> tuple[Path | None, str]:
    """클립 실제 파일 위치와 카테고리(clips/Good/Bad)를 반환.

    탐색 순서: runs/{job_id}/clips/ → highlights/Good/ → highlights/Bad/
    """
    # 경로 traversal 방지
    if "/" in clip_name or "\\" in clip_name or clip_name.startswith(".."):
        raise HTTPException(status_code=400, detail="잘못된 파일명")

    candidates = [
        (_job_dir(job_id) / "clips" / clip_name, "clips"),
        (HIGHLIGHTS_DIR / "Good" / clip_name, "Good"),
        (HIGHLIGHTS_DIR / "Bad" / clip_name, "Bad"),
    ]
    for p, cat in candidates:
        if p.exists():
            return p, cat
    return None, ""


def _augment_metadata_with_categories(metadata: dict[str, Any]) -> dict[str, Any]:
    """metadata.clips 각 항목에 대해 현재 파일 위치 카테고리를 추가."""
    job_id = metadata.get("job_id", "")
    categories: dict[str, str] = {}
    for clip in metadata.get("clips", []) or []:
        _, cat = _locate_clip(job_id, clip)
        categories[clip] = cat or "missing"
    metadata["clip_categories"] = categories
    return metadata


def _load_event_marks() -> dict[str, Any]:
    if not EVENT_MARKS_PATH.exists():
        return {}
    try:
        return json.loads(EVENT_MARKS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _save_event_marks(marks: dict[str, Any]) -> None:
    EVENT_MARKS_PATH.write_text(
        json.dumps(marks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _resolve_clip_anchor(metadata: dict[str, Any], clip_name: str) -> tuple[int | None, float]:
    """metadata에서 해당 clip_name의 anchor_frame과 fps를 찾는다."""
    clips = metadata.get("clips", []) or []
    frames = metadata.get("highlight_frames", []) or []
    fps = float(metadata.get("fps") or 30.0)

    # 1) 정렬 기반 매칭 (extract.py가 highlight_frames와 clips를 같은 순서로 저장)
    if clip_name in clips:
        idx = clips.index(clip_name)
        if idx < len(frames):
            return int(frames[idx]), fps

    # 2) trimmed 클립이면 원본 클립의 anchor 사용
    trimmed = metadata.get("trimmed", {}) or {}
    if clip_name in trimmed:
        base = trimmed[clip_name].get("from")
        if base and base in clips:
            idx = clips.index(base)
            if idx < len(frames):
                return int(frames[idx]), fps

    return None, fps


def _run_log_job(
    job_id: str,
    video_path: Path,
    log_data: list,
    second_half_start_sec: float,
    highlight_count: int,
) -> None:
    job_dir = _job_dir(job_id)
    clips_dir = job_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(job_id)
    metadata.update({"status": "running", "message": "로그 기반 분석 중..."})
    _save_metadata(job_id, metadata)

    result = run_log_pipeline(
        video_path=str(video_path),
        log_data=log_data,
        second_half_start_sec=second_half_start_sec,
        output_dir=str(clips_dir),
        target_count=highlight_count,
        model_path=str(MODEL_PATH),
    )

    clip_files = [Path(p).name for p in (result.clip_paths or [])]
    selected = {name: False for name in clip_files}

    # 클립별 영상 내 시간 구간 계산
    fps_val = float(result.fps or 30.0)
    clip_timestamps_by_name: dict[str, dict] = {}
    for meta in (result.events or []):
        if meta.get("source") == "log":
            name = meta["clip"]
            video_sec = float(meta.get("video_sec", 0))
            start_sec = max(0.0, video_sec - LOG_CLIP_BEFORE)
            end_sec = video_sec + LOG_CLIP_AFTER
            clip_timestamps_by_name[name] = {"start": round(start_sec, 1), "end": round(end_sec, 1)}

    log_names = set(clip_timestamps_by_name.keys())
    ai_clip_names = [n for n in clip_files if n not in log_names]
    for name, frame in zip(ai_clip_names, result.highlight_frames or []):
        anchor_sec = frame / fps_val
        start_sec = max(0.0, anchor_sec - CLIP_DURATION_BEFORE)
        end_sec = anchor_sec + CLIP_DURATION_AFTER
        clip_timestamps_by_name[name] = {"start": round(start_sec, 1), "end": round(end_sec, 1)}

    metadata.update({
        "status": "done" if result.success else "error",
        "message": result.message,
        "clips": clip_files,
        "selected": selected,
        "highlight_frames": result.highlight_frames,
        "fps": result.fps,
        "mode": "log_hybrid",
        "log_events": result.events,
        "clip_features": result.clip_features,
        "clip_feature_stats": result.clip_feature_stats,
        "clip_scores": result.clip_scores,
        "clip_timestamps": clip_timestamps_by_name,
        "finished_at": time.time(),
    })
    _save_metadata(job_id, metadata)


def _probe_dimensions(path: Path) -> tuple[int, int]:
    """영상 (width, height) px. 실패 시 (0, 0)."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", str(path)],
            check=True, capture_output=True, text=True,
        )
        w, h = r.stdout.strip().split("x")
        return int(w), int(h)
    except Exception:
        return 0, 0


def _make_playback_proxy(src: Path, out: Path, target_h: int = 720) -> bool:
    """리뷰 재생 전용 저화질 프록시 생성(720p·빠른 디코딩·무음).

    탐지는 원본으로, 클립 추출도 원본으로 진행 — 프록시는 브라우저에서 3·4배속 재생이
    부드럽도록 '보여주기'에만 쓴다(고해상도 원본은 고배속 디코딩이 안 됨). 박스 좌표는
    원본 픽셀 기준이라 프론트가 원본 W/H 로 스케일하므로 프록시 해상도는 무관.
    """
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-vf", f"scale=-2:{target_h}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
             "-an", "-movflags", "+faststart", str(out)],
            check=True, capture_output=True,
        )
        return out.exists()
    except Exception as e:
        print(f"[경고] 재생 프록시 생성 실패 → 원본 재생으로 폴백: {e}")
        return False


def _run_proxy_job(job_id: str) -> None:
    """기존 job 의 원본으로 재생 프록시만 생성(재탐지 없이). 메타에 proxy_file/W/H 기록."""
    meta = _load_metadata(job_id)
    vf = meta.get("video_file")
    src = UPLOADS_DIR / vf if vf else None
    if not src or not src.exists():
        meta["proxy_status"] = "error"; _save_metadata(job_id, meta); return
    vw, vh = _probe_dimensions(src)
    ok = _make_playback_proxy(src, _job_dir(job_id) / "proxy.mp4")
    meta = _load_metadata(job_id)  # 그새 바뀌었을 수 있어 재로드
    meta.update({"video_w": vw or meta.get("video_w", 0),
                 "video_h": vh or meta.get("video_h", 0),
                 "proxy_file": "proxy.mp4" if ok else None,
                 "proxy_status": "done" if ok else "error"})
    _save_metadata(job_id, meta)


def _run_player_detect_job(job_id: str, video_path: Path) -> None:
    """개인 클립: 영상 탐지·추적 → detections CSV 저장 (클립은 이후 /extract 에서)."""
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(job_id)
    metadata.update({"status": "running", "message": "선수·공 탐지 중... (수 분 소요)"})
    _save_metadata(job_id, metadata)

    result = run_player_detection(str(video_path), str(job_dir))

    # 리뷰 재생용 저화질 프록시(원본 높이>720 일 때만). 원본 W/H 는 박스 좌표 스케일용.
    vw, vh = _probe_dimensions(video_path)
    proxy_file = None
    if result.success and vh > 720:
        metadata.update({"message": "재생용 프록시 생성 중..."})
        _save_metadata(job_id, metadata)
        if _make_playback_proxy(video_path, job_dir / "proxy.mp4"):
            proxy_file = "proxy.mp4"

    metadata.update({
        "status": "done" if result.success else "error",
        "message": result.message,
        "mode": "player_clip",
        "fps": result.fps,
        "detection_file": result.detection_file,
        "n_frames": result.n_frames,
        "n_player_tracks": result.n_player_tracks,
        "video_file": Path(video_path).name,
        "proxy_file": proxy_file,
        "video_w": vw,
        "video_h": vh,
        "clips": [],
        "selected": {},
        "finished_at": time.time(),
    })
    _save_metadata(job_id, metadata)


def _run_job(job_id: str, video_path: Path, highlight_count: int) -> None:
    job_dir = _job_dir(job_id)
    clips_dir = job_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    metadata = _load_metadata(job_id)
    metadata.update({"status": "running", "message": "분석 중..."})
    _save_metadata(job_id, metadata)

    result = run_highlight_pipeline(
        video_path=str(video_path),
        model_path=str(MODEL_PATH),
        output_dir=str(clips_dir),
        highlight_count=highlight_count
    )

    # [수정] 접두어가 붙은 CSV 파일들을 찾아 복사하고 메타데이터에 기록
    score_file = "xh_scores.csv"
    detection_file = "detections.csv"
    
    for f in clips_dir.glob("*_xh_scores.csv"):
        score_file = f.name
        shutil.copy2(f, job_dir / f.name)
    for f in clips_dir.glob("*_detections.csv"):
        detection_file = f.name
        shutil.copy2(f, job_dir / f.name)
    # clip_mapping.csv 도 run_dir 로 복사 (make_dataset.py / 재학습 파이프라인 필수)
    for f in clips_dir.glob("*_clip_mapping.csv"):
        shutil.copy2(f, job_dir / f.name)

    # 기본 이름 파일도 체크 (하위 호환성)
    if (clips_dir / "xh_scores.csv").exists() and not (job_dir / "xh_scores.csv").exists():
        shutil.copy2(clips_dir / "xh_scores.csv", job_dir / "xh_scores.csv")
    if (clips_dir / "detections.csv").exists() and not (job_dir / "detections.csv").exists():
        shutil.copy2(clips_dir / "detections.csv", job_dir / "detections.csv")
    if (clips_dir / "clip_mapping.csv").exists() and not (job_dir / "clip_mapping.csv").exists():
        shutil.copy2(clips_dir / "clip_mapping.csv", job_dir / "clip_mapping.csv")

    clip_files = [Path(p).name for p in (result.clip_paths or [])]
    clip_features = result.clip_features or []
    clip_feature_map = {
        name: feature for name, feature in zip(clip_files, clip_features)
    }
    clip_feature_stats = {}
    if result.clip_feature_stats:
        frame_stats = result.clip_feature_stats
        for name, frame in zip(clip_files, result.highlight_frames or []):
            if frame in frame_stats:
                clip_feature_stats[name] = frame_stats[frame]
    selected = {name: False for name in clip_files}

    clip_scores_by_name: dict[str, float] = {}
    if result.clip_scores and result.highlight_frames:
        for name, frame in zip(clip_files, result.highlight_frames):
            score = result.clip_scores.get(frame)
            if score is not None:
                clip_scores_by_name[name] = score

    fps_val = float(result.fps or 30.0)
    clip_timestamps_by_name: dict[str, dict] = {}
    for name, frame in zip(clip_files, result.highlight_frames or []):
        anchor_sec = frame / fps_val
        start_sec = max(0.0, anchor_sec - CLIP_DURATION_BEFORE)
        end_sec = anchor_sec + CLIP_DURATION_AFTER
        clip_timestamps_by_name[name] = {"start": round(start_sec, 1), "end": round(end_sec, 1)}

    metadata.update(
        {
            "status": "done" if result.success else "error",
            "message": result.message,
            "clips": clip_files,
            "clip_features": clip_feature_map,
            "clip_feature_stats": clip_feature_stats,
            "clip_scores": clip_scores_by_name,
            "clip_timestamps": clip_timestamps_by_name,
            "selected": selected,
            "highlight_frames": result.highlight_frames or [],
            "fps": result.fps,
            "score_file": score_file,
            "detection_file": detection_file,
            "finished_at": time.time(),
        }
    )
    _save_metadata(job_id, metadata)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_path = BASE_DIR / "app" / "static" / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/jobs")
def list_jobs() -> dict[str, Any]:
    """기존 runs/ 폴더를 스캔해서 모든 job 목록 반환 (최신순)."""
    jobs: list[dict[str, Any]] = []
    if not RUNS_DIR.exists():
        return {"jobs": []}

    for job_dir in RUNS_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        meta_path = job_dir / "metadata.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        clips = meta.get("clips", []) or []
        created_at = meta.get("created_at") or meta_path.stat().st_mtime
        jobs.append({
            "job_id": meta.get("job_id", job_dir.name),
            "status": meta.get("status", "unknown"),
            "created_at": created_at,
            "clip_count": len(clips),
            "highlight_count": meta.get("highlight_count"),
            "source_filename": meta.get("source_filename"),
        })

    # 최신순 정렬
    jobs.sort(key=lambda x: x.get("created_at") or 0, reverse=True)
    return {"jobs": jobs}


@app.post("/api/jobs")
async def create_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    highlight_count: int = Form(DEFAULT_HIGHLIGHT_COUNT),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="업로드할 파일이 없습니다.")

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".mp4"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    saved_video = UPLOADS_DIR / f"{job_id}{ext}"
    with saved_video.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    saved_video = _normalize_for_analysis(saved_video)  # MTS 등만 mp4 변환

    metadata = {
        "job_id": job_id,
        "status": "queued",
        "message": "대기 중",
        "clips": [],
        "selected": {},
        "created_at": time.time(),
        "highlight_count": highlight_count,
        "source_filename": file.filename,
    }
    _save_metadata(job_id, metadata)

    background_tasks.add_task(_run_job, job_id, saved_video, highlight_count)
    return {"job_id": job_id, "status": "queued"}


MATCHDAY_API_BASE = os.getenv("MATCHDAY_API_BASE", "https://api.matchday-planner.com/api/v1").rstrip("/")


def _fetch_match_history(match_id: str) -> list:
    """matchday-planner 경기 로그(history) 조회 → 이벤트 배열만 반환.

    응답은 {"status":200,"data":[...]} 형태라 data 배열만 꺼낸다.
    """
    url = f"{MATCHDAY_API_BASE}/matches/{match_id}/history"
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        payload = resp.json()
    except Exception as ex:
        raise HTTPException(status_code=502, detail=f"경기 로그 조회 실패 (match_id={match_id}): {ex}")
    data = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="경기 로그 응답에 data 배열이 없습니다.")
    return data


@app.post("/api/log-jobs")
async def create_log_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    log_data: str = Form("[]"),
    match_id: str = Form(""),
    second_half_start_min: int = Form(25),
    second_half_start_sec_extra: int = Form(0),
    highlight_count: int = Form(DEFAULT_HIGHLIGHT_COUNT),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="업로드할 파일이 없습니다.")

    if match_id.strip():
        events = _fetch_match_history(match_id.strip())
    else:
        try:
            parsed = json.loads(log_data)
            # API 응답 전체({status, data:[...]}) 또는 배열 자체 모두 허용
            if isinstance(parsed, dict):
                events = parsed.get("data", [])
            elif isinstance(parsed, list):
                events = parsed
            else:
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="log_data가 유효한 JSON이 아닙니다.")

    second_half_start_sec = second_half_start_min * 60 + second_half_start_sec_extra

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".mp4"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    saved_video = UPLOADS_DIR / f"{job_id}{ext}"
    with saved_video.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    saved_video = _normalize_for_analysis(saved_video)  # MTS 등만 mp4 변환

    metadata = {
        "job_id": job_id,
        "status": "queued",
        "message": "대기 중",
        "clips": [],
        "selected": {},
        "created_at": time.time(),
        "highlight_count": highlight_count,
        "source_filename": file.filename,
        "mode": "log_hybrid",
        "second_half_start_sec": second_half_start_sec,
    }
    _save_metadata(job_id, metadata)

    background_tasks.add_task(
        _run_log_job, job_id, saved_video, events,
        second_half_start_sec, highlight_count,
    )
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/player-jobs")
async def create_player_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """개인 클립: 영상 업로드 → 선수·공 탐지·추적(백그라운드). 이후 선수 지정 → /extract."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="업로드할 영상 파일이 없습니다.")

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    ext = Path(file.filename).suffix or ".mp4"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    saved_video = UPLOADS_DIR / f"{job_id}{ext}"
    with saved_video.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    saved_video = _normalize_for_analysis(saved_video)

    metadata = {
        "job_id": job_id,
        "status": "queued",
        "message": "대기 중",
        "clips": [],
        "selected": {},
        "created_at": time.time(),
        "source_filename": file.filename,
        "mode": "player_clip",
        "video_file": saved_video.name,
    }
    _save_metadata(job_id, metadata)

    background_tasks.add_task(_run_player_detect_job, job_id, saved_video)
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/player-jobs/{job_id}/video")
def serve_player_video(job_id: str):
    """리뷰 재생용 영상 서빙 — 프록시(저화질)가 있으면 그것, 없으면 원본.

    고배속 디코딩이 부드럽도록 프록시 우선. 탐지·클립추출은 원본을 쓰므로 화질 영향 없음.
    """
    _require_job(job_id)
    meta = _load_metadata(job_id)
    proxy = meta.get("proxy_file")
    path = None
    if proxy:
        p = _job_dir(job_id) / proxy
        if p.exists():
            path = p
    if path is None:
        vf = meta.get("video_file")
        path = UPLOADS_DIR / vf if vf else None
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="영상 파일을 찾을 수 없습니다.")
    suffix = path.suffix.lower()
    media = "video/mp4" if suffix in (".mp4", ".m4v", ".mov") else f"video/{suffix.lstrip('.')}"
    return FileResponse(str(path), media_type=media, filename=path.name)


class PlayerExtractBody(BaseModel):
    track_ids: list[int] = []
    # 시간 구간에 묶인 선수 지정 [track_id, from_sec, to_sec]. 같은 id 가 화면전환 후
    # 다른 선수에게 재사용돼도 그 구간만 인정 → 한 선수가 여러 id 를 거쳐도 누적된다.
    track_windows: list[list[float]] = []
    # 공 미탐지 등으로 자동 이벤트가 없는 구간을 사용자가 직접 찍은 관여 시각(초) 목록.
    direct_marks: list[float] = []
    pad_before: float = PLAYER_PAD_BEFORE
    pad_after: float = PLAYER_PAD_AFTER
    exclude_intervals: list[list[float]] = []  # 신원검증서 제외한 (시작,끝)초 구간들


def _load_player_det(job_id: str) -> tuple[pd.DataFrame, float, str]:
    """개인클립 job 의 (detections df, fps, video_file) 로드. 없으면 400."""
    meta = _load_metadata(job_id)
    det_path = _job_dir(job_id) / meta.get("detection_file", PLAYER_DETECTION_FILE)
    vf = meta.get("video_file")
    if not det_path.exists() or not vf:
        raise HTTPException(status_code=400, detail="탐지 결과가 없습니다. 먼저 탐지를 완료하세요.")
    return pd.read_csv(det_path), float(meta.get("fps") or 30.0), vf


def _player_track_windows(body: PlayerExtractBody, fps: float):
    """body 의 초 단위 구간 → (tid, from_frame, to_frame) 목록. 없으면 None."""
    return [(int(t), float(a) * fps, float(b) * fps)
            for t, a, b in body.track_windows] or None


@app.post("/api/player-jobs/{job_id}/proxy")
def make_player_proxy(job_id: str, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """기존 job 에 재생 프록시를 즉석 생성(재탐지 X, ffmpeg 만). 백그라운드 실행 → 폴링."""
    _require_job(job_id)
    meta = _load_metadata(job_id)
    if meta.get("proxy_file"):
        return {"status": "done", "proxy_file": meta["proxy_file"]}
    if meta.get("proxy_status") == "running":
        return {"status": "running"}
    meta["proxy_status"] = "running"
    _save_metadata(job_id, meta)
    background_tasks.add_task(_run_proxy_job, job_id)
    return {"status": "running"}


@app.get("/api/player-jobs/{job_id}/events")
def list_player_possession_events(job_id: str) -> dict[str, Any]:
    """영상 전체의 공 보유 이벤트(클립 후보) 열거 — 옵션 A 이벤트 리뷰용."""
    _require_job(job_id)
    det_df, fps, _ = _load_player_det(job_id)
    events = compute_possession_events(det_df, fps)
    return {"job_id": job_id, "fps": fps, "event_count": len(events), "events": events}


@app.post("/api/player-jobs/{job_id}/preview")
def preview_player_job_clips(job_id: str, body: PlayerExtractBody) -> dict[str, Any]:
    """컷 없이 예상 클립 수·구간만 계산(미리보기). ffmpeg 미실행이라 빠름."""
    _require_job(job_id)
    if not body.track_ids and not body.track_windows and not body.direct_marks:
        return {"job_id": job_id, "clip_count": 0, "segments": []}
    det_df, fps, _ = _load_player_det(job_id)
    segments = compute_player_segments(
        det_df, fps, body.track_ids,
        pad_before=body.pad_before, pad_after=body.pad_after,
        exclude_intervals=[tuple(iv) for iv in body.exclude_intervals],
        track_windows=_player_track_windows(body, fps),
        extra_involve_secs=list(body.direct_marks),
    )
    total = round(sum(s["end"] - s["start"] for s in segments), 1)
    return {"job_id": job_id, "clip_count": len(segments),
            "total_seconds": total, "segments": segments}


@app.post("/api/player-jobs/{job_id}/extract")
def extract_player_job_clips(job_id: str, body: PlayerExtractBody) -> dict[str, Any]:
    """지정한 선수 track_id 들이 공에 관여한 구간만 클립으로 추출."""
    _require_job(job_id)
    if not body.track_ids and not body.track_windows and not body.direct_marks:
        raise HTTPException(status_code=400, detail="선수를 한 명 이상 지정하세요.")
    det_df, fps, vf = _load_player_det(job_id)
    job_dir = _job_dir(job_id)
    meta = _load_metadata(job_id)
    # 초 단위 구간 → 프레임 구간으로 변환해 전달
    track_windows = _player_track_windows(body, fps)
    clip_paths, seg_meta = extract_player_clips(
        str(UPLOADS_DIR / vf), det_df, fps, body.track_ids, str(job_dir),
        pad_before=body.pad_before, pad_after=body.pad_after,
        exclude_intervals=[tuple(iv) for iv in body.exclude_intervals],
        track_windows=track_windows,
        extra_involve_secs=list(body.direct_marks),
    )
    clip_files = [Path(p).name for p in clip_paths]
    meta.update({
        "clips": clip_files,
        "selected": {n: False for n in clip_files},
        "clip_timestamps": {m["clip"]: {"start": m["start"], "end": m["end"]} for m in seg_meta},
        "player_segments": seg_meta,
        "extracted_track_ids": body.track_ids or sorted({int(w[0]) for w in body.track_windows}),
    })
    _save_metadata(job_id, meta)
    return {"job_id": job_id, "clip_count": len(clip_files),
            "clips": clip_files, "segments": seg_meta}


# ──────────────────────────────────────────────
# 전반/후반 분리 업로드 (concat → 단일 파이프라인)
# ──────────────────────────────────────────────

def _get_video_duration(video_path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
        capture_output=True, text=True, check=True,
    )
    return float(r.stdout.strip())


def _probe_duration(path: Path) -> float:
    """ffprobe 로 영상 길이(초)를 반환. 실패 시 0.0."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            check=True, capture_output=True,
        )
        return float(r.stdout.decode("utf-8", "ignore").strip())
    except (subprocess.CalledProcessError, ValueError):
        return 0.0


def _concat_videos(video_paths: list[Path], output_path: Path) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        for p in video_paths:
            f.write(f"file '{p}'\n")
        list_file = f.name
    subprocess.run(
        ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
         "-i", list_file, "-c", "copy", str(output_path)],
        check=True, capture_output=True,
    )
    Path(list_file).unlink(missing_ok=True)


# OpenCV/extract 가 fps 를 잘못 읽는 카메라 트랜스포트 스트림(AVCHD 등) 형식.
# 이런 입력은 인터레이스·가변 프레임레이트가 많아 클립 타임스탬프가 어긋난다.
TRANSCODE_EXTS = {".mts", ".m2ts", ".m2t", ".ts", ".mod", ".tod"}


def _normalize_for_analysis(src: Path) -> Path:
    """MTS 등 fps/인터레이스 이슈가 있는 입력만 분석용 mp4(고정 프레임레이트·디인터레이스)로 변환.

    표준 형식(mp4/mov 등)은 변환 없이 원본 경로를 그대로 반환(추가 시간 0).
    변환 성공 시 원본 파일은 삭제하고 mp4 경로 반환 → 분석·브라우저 재생·서빙을 mp4 로 통일.
    변환 실패 시에는 경고만 남기고 원본 경로로 폴백.
    """
    if src.suffix.lower() not in TRANSCODE_EXTS:
        return src
    out = src.with_suffix(".mp4")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(src),
             "-vf", "yadif=0", "-r", "30",          # 디인터레이스 + 고정 30fps
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-c:a", "aac", "-movflags", "+faststart", str(out)],
            check=True, capture_output=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"[경고] 입력 정규화(트랜스코딩) 실패 → 원본으로 분석 진행: {src.name} ({e})")
        return src
    src.unlink(missing_ok=True)
    return out


@app.post("/api/jobs/split")
async def create_split_job(
    background_tasks: BackgroundTasks,
    first_half: UploadFile = File(...),
    second_half: UploadFile = File(...),
    highlight_count: int = Form(DEFAULT_HIGHLIGHT_COUNT),
) -> dict[str, Any]:
    for f in (first_half, second_half):
        if not f.filename:
            raise HTTPException(status_code=400, detail="파일을 모두 선택하세요.")

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    ext1 = Path(first_half.filename).suffix or ".mp4"
    ext2 = Path(second_half.filename).suffix or ".mp4"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path1 = UPLOADS_DIR / f"{job_id}_1{ext1}"
    path2 = UPLOADS_DIR / f"{job_id}_2{ext2}"
    with path1.open("wb") as buf: shutil.copyfileobj(first_half.file, buf)
    with path2.open("wb") as buf: shutil.copyfileobj(second_half.file, buf)
    path1 = _normalize_for_analysis(path1)  # MTS 등만 mp4 변환
    path2 = _normalize_for_analysis(path2)

    second_half_start_sec = _get_video_duration(path1)
    combined = UPLOADS_DIR / f"{job_id}_combined.mp4"
    _concat_videos([path1, path2], combined)

    source_name = f"{Path(first_half.filename).stem} + {Path(second_half.filename).stem}"
    metadata = {
        "job_id": job_id,
        "status": "queued",
        "message": "대기 중",
        "clips": [],
        "selected": {},
        "created_at": time.time(),
        "highlight_count": highlight_count,
        "source_filename": source_name,
        "second_half_start_sec": round(second_half_start_sec, 1),
    }
    _save_metadata(job_id, metadata)

    background_tasks.add_task(_run_job, job_id, combined, highlight_count)
    return {"job_id": job_id, "status": "queued"}


@app.post("/api/log-jobs/split")
async def create_split_log_job(
    background_tasks: BackgroundTasks,
    first_half: UploadFile = File(...),
    second_half: UploadFile = File(...),
    log_data: str = Form("[]"),
    match_id: str = Form(""),
    highlight_count: int = Form(DEFAULT_HIGHLIGHT_COUNT),
) -> dict[str, Any]:
    for f in (first_half, second_half):
        if not f.filename:
            raise HTTPException(status_code=400, detail="파일을 모두 선택하세요.")

    if match_id.strip():
        events = _fetch_match_history(match_id.strip())
    else:
        try:
            parsed = json.loads(log_data)
            events = parsed.get("data", []) if isinstance(parsed, dict) else parsed
            if not isinstance(events, list):
                raise ValueError
        except (json.JSONDecodeError, ValueError):
            raise HTTPException(status_code=400, detail="log_data가 유효한 JSON이 아닙니다.")

    job_id = uuid.uuid4().hex
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    ext1 = Path(first_half.filename).suffix or ".mp4"
    ext2 = Path(second_half.filename).suffix or ".mp4"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    path1 = UPLOADS_DIR / f"{job_id}_1{ext1}"
    path2 = UPLOADS_DIR / f"{job_id}_2{ext2}"
    with path1.open("wb") as buf: shutil.copyfileobj(first_half.file, buf)
    with path2.open("wb") as buf: shutil.copyfileobj(second_half.file, buf)
    path1 = _normalize_for_analysis(path1)  # MTS 등만 mp4 변환
    path2 = _normalize_for_analysis(path2)

    second_half_start_sec = _get_video_duration(path1)
    combined = UPLOADS_DIR / f"{job_id}_combined.mp4"
    _concat_videos([path1, path2], combined)

    source_name = f"{Path(first_half.filename).stem} + {Path(second_half.filename).stem}"
    metadata = {
        "job_id": job_id,
        "status": "queued",
        "message": "대기 중",
        "clips": [],
        "selected": {},
        "created_at": time.time(),
        "highlight_count": highlight_count,
        "source_filename": source_name,
        "mode": "log_hybrid",
        "second_half_start_sec": round(second_half_start_sec, 1),
    }
    _save_metadata(job_id, metadata)

    background_tasks.add_task(
        _run_log_job, job_id, combined, events,
        second_half_start_sec, highlight_count,
    )
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    _require_job(job_id)
    return _augment_metadata_with_categories(_load_metadata(job_id))


@app.get("/api/jobs/{job_id}/clips/{clip_name}/file")
def serve_clip_file(job_id: str, clip_name: str):
    """클립 파일을 실제 위치에서 찾아 스트리밍.

    clips/ → highlights/Good/ → highlights/Bad/ 순으로 탐색.
    """
    _require_job(job_id)
    path, _ = _locate_clip(job_id, clip_name)
    if path is None:
        raise HTTPException(status_code=404, detail=f"클립 파일 없음: {clip_name}")
    return FileResponse(str(path), media_type="video/mp4", filename=clip_name)


@app.post("/api/jobs/{job_id}/clips/{clip_name}/classify")
def classify_clip(job_id: str, clip_name: str, category: str) -> dict[str, Any]:
    """클립을 Good / Bad / uncategorized(=clips) 로 이동.

    - Good / Bad: highlights/{Good|Bad}/ 로 move
    - clips: 다시 runs/{job_id}/clips/ 로 되돌림 (분류 취소)
    """
    _require_job(job_id)
    target = str(category).strip().capitalize()
    if target not in {"Good", "Bad", "Clips"}:
        raise HTTPException(status_code=400, detail="category 는 Good / Bad / clips 중 하나")

    src, current_cat = _locate_clip(job_id, clip_name)
    if src is None:
        raise HTTPException(status_code=404, detail=f"클립 없음: {clip_name}")

    if target == "Clips":
        dst_dir = _job_dir(job_id) / "clips"
        dst_dir.mkdir(exist_ok=True, parents=True)
        dst = dst_dir / clip_name
        new_cat = "clips"
    else:
        dst_dir = HIGHLIGHTS_DIR / target
        dst_dir.mkdir(exist_ok=True, parents=True)
        dst = dst_dir / clip_name
        new_cat = target

    # 이미 해당 위치에 있으면 no-op
    if src.resolve() == dst.resolve():
        return {"ok": True, "category": new_cat, "moved": False}

    # 목적지에 이미 파일이 있으면 덮어쓰기 방지 → 기존 파일 제거
    if dst.exists():
        dst.unlink()
    shutil.move(str(src), str(dst))
    return {"ok": True, "category": new_cat, "moved": True, "from": current_cat, "to": new_cat}


@app.post("/api/jobs/{job_id}/clips/{clip_name}/select")
def select_clip(job_id: str, clip_name: str, selected: bool) -> dict[str, Any]:
    _require_job(job_id)
    clip_path, _ = _locate_clip(job_id, clip_name)
    if clip_path is None:
        raise HTTPException(status_code=404, detail="클립을 찾을 수 없습니다.")

    metadata = _load_metadata(job_id)
    metadata.setdefault("selected", {})[clip_name] = bool(selected)
    _save_metadata(job_id, metadata)
    return {"ok": True, "selected": metadata["selected"]}


@app.delete("/api/jobs/{job_id}/clips/{clip_name}")
def delete_clip(job_id: str, clip_name: str) -> dict[str, Any]:
    _require_job(job_id)
    clip_path, _ = _locate_clip(job_id, clip_name)
    if clip_path is None:
        raise HTTPException(status_code=404, detail="클립을 찾을 수 없습니다.")

    clip_path.unlink()
    metadata = _load_metadata(job_id)
    metadata["clips"] = [name for name in metadata.get("clips", []) if name != clip_name]
    metadata.setdefault("selected", {}).pop(clip_name, None)
    _save_metadata(job_id, metadata)
    return {"ok": True, "clips": metadata["clips"]}


@app.post("/api/jobs/{job_id}/clips/{clip_name}/trim")
def trim_clip(job_id: str, clip_name: str, start: float = 0.0, end: float = 0.0) -> dict[str, Any]:
    _require_job(job_id)
    clip_path, _ = _locate_clip(job_id, clip_name)
    if clip_path is None:
        raise HTTPException(status_code=404, detail="클립을 찾을 수 없습니다.")

    if start < 0 or end <= 0 or end <= start:
        raise HTTPException(status_code=400, detail="끝 시간이 시작 시간보다 커야 합니다.")

    # trimmed 파일은 원래 위치 옆에 저장 (Good/Bad 에서 잘랐으면 그 폴더에 생성).
    # 같은 소스를 여러 구간으로 컷 편집할 때 파일명이 겹치지 않도록 고유화한다
    # (외부 영상을 5분/20분 두 군데 자르면 둘 다 trimmed_ext_xxx.mp4 가 되어
    #  뒤 컷이 앞 컷을 덮어쓰던 버그 방지).
    base_trim = f"trimmed_{clip_name}"
    trimmed_path = clip_path.with_name(base_trim)
    if trimmed_path.exists():
        stem, suffix = Path(base_trim).stem, Path(base_trim).suffix
        i = 1
        while trimmed_path.exists():
            trimmed_path = clip_path.with_name(f"{stem}_{i}{suffix}")
            i += 1

    # 실제 길이를 ffprobe 로 확인해 end 클램프 (외부 VFR 영상도 정확)
    src_duration = _probe_duration(clip_path)
    end_time = min(end, src_duration) if src_duration else end
    if start >= end_time:
        raise HTTPException(status_code=400, detail="잘못된 시간 범위입니다.")

    # ffmpeg 재인코딩 트림: moviepy 는 가변 프레임레이트(VFR) 외부 영상의 프레임 간격을
    # 일정하다고 가정해 출력 길이가 짧아지고 → 배속처럼 보임. ffmpeg 는 타임스탬프를
    # 그대로 보존하며 CFR 로 정규화하므로 배속 문제가 없다.
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", str(clip_path),
             "-t", f"{end_time - start:.3f}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
             "-c:a", "aac", "-vsync", "cfr", "-movflags", "+faststart",
             str(trimmed_path)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode("utf-8", "ignore")[-500:] if e.stderr else ""
        raise HTTPException(status_code=500, detail=f"트림 실패: {stderr}") from e

    base_name = clip_name
    while base_name.startswith("trimmed_"):
        base_name = base_name[len("trimmed_"):]
    is_external = base_name.startswith("ext_")

    metadata = _load_metadata(job_id)

    # 외부 영상이지만 후보 그리드에 등록되지 않은 경우(편집기 타임라인 전용):
    # 분석 metadata에 등록하지 않고 in-place 교체용으로만 반환.
    # → 그리드에 등록된 외부 클립(register=true)은 아래 일반 경로로 metadata.clips 에 추가.
    if is_external and clip_name not in (metadata.get("clips") or []):
        return {"ok": True, "clip": trimmed_path.name, "video_start": round(start, 1), "video_end": round(end_time, 1)}

    metadata.setdefault("clips", []).append(trimmed_path.name)
    metadata.setdefault("selected", {})[trimmed_path.name] = False
    metadata.setdefault("trimmed", {})[trimmed_path.name] = {"from": clip_name, "start": start, "end": end_time}

    # 원본 영상 기준 타임스탬프: 부모 클립의 video_start + 트림 오프셋
    orig_ts = (metadata.get("clip_timestamps") or {}).get(clip_name) or {}
    orig_video_start = float(orig_ts.get("start", 0.0))
    metadata.setdefault("clip_timestamps", {})[trimmed_path.name] = {
        "start": round(orig_video_start + start, 1),
        "end":   round(orig_video_start + end_time, 1),
    }

    _save_metadata(job_id, metadata)
    return {"ok": True, "clip": trimmed_path.name, "video_start": round(orig_video_start + start, 1), "video_end": round(orig_video_start + end_time, 1)}


@app.post("/api/jobs/{job_id}/clips/{clip_name}/event")
def set_event_frame(
    job_id: str,
    clip_name: str,
    event_time_in_clip: float = Form(...),
    event_type: str = Form("unknown"),
) -> dict[str, Any]:
    """사용자가 클립 내에서 '실제 이벤트가 일어난 시점(초)'을 지정.

    클립 시작 시각 = anchor_sec - CLIP_DURATION_BEFORE 이므로,
    원본 영상의 event 프레임 번호는
        event_frame_original = (anchor_sec - 15 + event_time_in_clip) * fps
    로 계산해 event_marks.json에 저장한다.
    """
    _require_job(job_id)
    clip_path, _ = _locate_clip(job_id, clip_name)
    if clip_path is None:
        raise HTTPException(status_code=404, detail="클립을 찾을 수 없습니다.")
    if event_time_in_clip < 0:
        raise HTTPException(status_code=400, detail="event_time_in_clip은 0 이상이어야 합니다.")

    # event_type 정규화 (허용되지 않는 값은 'unknown')
    event_type_norm = str(event_type).strip().lower() or "unknown"
    if event_type_norm not in EVENT_TYPE_SET:
        event_type_norm = "unknown"

    metadata = _load_metadata(job_id)
    anchor_frame, fps = _resolve_clip_anchor(metadata, clip_name)
    if anchor_frame is None:
        raise HTTPException(status_code=400, detail="클립의 anchor 정보를 찾을 수 없습니다.")
    if fps <= 0:
        fps = 30.0

    anchor_sec = anchor_frame / fps
    clip_start_sec = max(0.0, anchor_sec - CLIP_DURATION_BEFORE)
    event_frame_original = int(round((clip_start_sec + event_time_in_clip) * fps))

    marks = _load_event_marks()
    marks[clip_name] = {
        "clip_name": clip_name,
        "job_id": job_id,
        "anchor_frame": int(anchor_frame),
        "fps": float(fps),
        "clip_start_sec": float(clip_start_sec),
        "event_time_in_clip_sec": float(event_time_in_clip),
        "event_frame_original": int(event_frame_original),
        "event_type": event_type_norm,
        "marked_at": time.time(),
    }
    _save_event_marks(marks)

    # 편의: 해당 job의 metadata에도 event_marks 요약 저장
    metadata.setdefault("event_marks", {})[clip_name] = {
        "event_frame_original": int(event_frame_original),
        "event_time_in_clip_sec": float(event_time_in_clip),
        "event_type": event_type_norm,
    }
    _save_metadata(job_id, metadata)

    return {"ok": True, "mark": marks[clip_name]}


@app.get("/api/jobs/{job_id}/clips/{clip_name}/event")
def get_event_frame(job_id: str, clip_name: str) -> dict[str, Any]:
    _require_job(job_id)
    marks = _load_event_marks()
    mark = marks.get(clip_name)
    if mark is None:
        raise HTTPException(status_code=404, detail="이벤트 프레임이 지정되지 않았습니다.")
    return mark


@app.delete("/api/jobs/{job_id}/clips/{clip_name}/event")
def delete_event_frame(job_id: str, clip_name: str) -> dict[str, Any]:
    _require_job(job_id)
    marks = _load_event_marks()
    if clip_name in marks:
        del marks[clip_name]
        _save_event_marks(marks)
        metadata = _load_metadata(job_id)
        if "event_marks" in metadata and clip_name in metadata["event_marks"]:
            del metadata["event_marks"][clip_name]
            _save_metadata(job_id, metadata)
    return {"ok": True}


class OverlayListBody(BaseModel):
    overlays: list[OverlayDef] = []


class BgmTracksBody(BaseModel):
    bgm_tracks: list["BgmTrack"] = []


@app.get("/api/jobs/{job_id}/overlays")
def get_editor_overlays(job_id: str) -> dict[str, Any]:
    """편집기 오버레이 트랙 조회 (합쳐진 영상 전체 기준)."""
    _require_job(job_id)
    metadata = _load_metadata(job_id)
    return {"overlays": metadata.get("editor_overlays", []) or []}


@app.post("/api/jobs/{job_id}/overlays")
def set_editor_overlays(job_id: str, body: OverlayListBody) -> dict[str, Any]:
    """편집기 오버레이 트랙 저장."""
    _require_job(job_id)
    metadata = _load_metadata(job_id)
    metadata["editor_overlays"] = [ov.model_dump() for ov in body.overlays]
    _save_metadata(job_id, metadata)
    return {"ok": True}


@app.get("/api/jobs/{job_id}/bgm-tracks")
def get_editor_bgm_tracks(job_id: str) -> dict[str, Any]:
    """편집기 BGM 트랙 조회."""
    _require_job(job_id)
    metadata = _load_metadata(job_id)
    return {"bgm_tracks": metadata.get("editor_bgm_tracks", []) or []}


@app.post("/api/jobs/{job_id}/bgm-tracks")
def set_editor_bgm_tracks(job_id: str, body: BgmTracksBody) -> dict[str, Any]:
    """편집기 BGM 트랙 저장."""
    _require_job(job_id)
    metadata = _load_metadata(job_id)
    metadata["editor_bgm_tracks"] = [t.model_dump() for t in body.bgm_tracks]
    _save_metadata(job_id, metadata)
    return {"ok": True}


# ── 편집기 프로젝트 저장/불러오기 (타임라인 순서 + 오버레이 + BGM + 설정) ──────
EDITOR_PROJECTS_PATH = BASE_DIR / "editor_projects.json"


def _load_editor_projects() -> dict[str, Any]:
    if EDITOR_PROJECTS_PATH.exists():
        try:
            return json.loads(EDITOR_PROJECTS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_editor_projects(data: dict[str, Any]) -> None:
    EDITOR_PROJECTS_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


class EditorProjectBody(BaseModel):
    name: str
    job_id: str | None = None
    timeline: list[dict] = []
    overlays: list[dict] = []
    bgm_tracks: list[dict] = []
    transition_sec: float = 0.5
    audio_volume: float = 1.0


@app.get("/api/editor/projects")
def list_editor_projects() -> dict[str, Any]:
    """저장된 편집 프로젝트 목록 (최근 수정순)."""
    data = _load_editor_projects()
    out = [
        {
            "id": pid,
            "name": p.get("name", "(이름없음)"),
            "created_at": p.get("created_at"),
            "updated_at": p.get("updated_at"),
            "item_count": len(p.get("timeline", []) or []),
            "overlay_count": len(p.get("overlays", []) or []),
        }
        for pid, p in data.items()
    ]
    out.sort(key=lambda x: x.get("updated_at") or 0, reverse=True)
    return {"projects": out}


@app.post("/api/editor/projects")
def save_editor_project(body: EditorProjectBody) -> dict[str, Any]:
    """편집 프로젝트 저장. 같은 이름이 있으면 덮어쓰기, 없으면 새로 생성."""
    data = _load_editor_projects()
    name = (body.name or "").strip() or "무제 프로젝트"
    pid = next((k for k, v in data.items() if v.get("name") == name), None)
    now = time.time()
    created = data[pid].get("created_at", now) if pid else now
    if pid is None:
        pid = uuid.uuid4().hex
    data[pid] = {
        "name": name,
        "created_at": created,
        "updated_at": now,
        "job_id": body.job_id,
        "timeline": body.timeline,
        "overlays": body.overlays,
        "bgm_tracks": body.bgm_tracks,
        "transition_sec": body.transition_sec,
        "audio_volume": body.audio_volume,
    }
    _save_editor_projects(data)
    return {"ok": True, "id": pid, "name": name}


@app.get("/api/editor/projects/{project_id}")
def get_editor_project(project_id: str) -> dict[str, Any]:
    """편집 프로젝트 전체 조회."""
    data = _load_editor_projects()
    p = data.get(project_id)
    if not p:
        raise HTTPException(status_code=404, detail="프로젝트를 찾을 수 없습니다.")
    return {"id": project_id, **p}


@app.delete("/api/editor/projects/{project_id}")
def delete_editor_project(project_id: str) -> dict[str, Any]:
    """편집 프로젝트 삭제."""
    data = _load_editor_projects()
    if project_id in data:
        del data[project_id]
        _save_editor_projects(data)
    return {"ok": True}


# ── 오버레이 이미지 (점수판/득점자 템플릿) ─────────────────
@app.post("/api/jobs/{job_id}/overlay-images")
async def upload_overlay_image(job_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """오버레이 이미지(점수판/득점자 등) 업로드."""
    _require_job(job_id)
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")
    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
    ov_dir = _job_dir(job_id) / "overlay_images"
    ov_dir.mkdir(exist_ok=True, parents=True)
    saved = ov_dir / safe_name
    with saved.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "name": safe_name}


@app.get("/api/jobs/{job_id}/overlay-images")
def list_overlay_images(job_id: str) -> dict[str, Any]:
    """오버레이 이미지 목록."""
    _require_job(job_id)
    ov_dir = _job_dir(job_id) / "overlay_images"
    if not ov_dir.exists():
        return {"images": []}
    images = sorted(f.name for f in ov_dir.iterdir() if f.is_file())
    return {"images": images}


@app.delete("/api/jobs/{job_id}/overlay-images/{filename}")
def delete_overlay_image(job_id: str, filename: str) -> dict[str, Any]:
    """오버레이 이미지 삭제."""
    _require_job(job_id)
    if "/" in filename or "\\" in filename or filename.startswith(".."):
        raise HTTPException(status_code=400, detail="잘못된 파일명")
    p = _job_dir(job_id) / "overlay_images" / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="이미지 없음")
    p.unlink()
    return {"ok": True}


@app.get("/api/event_marks")
def list_event_marks() -> dict[str, Any]:
    return _load_event_marks()


@app.get("/api/jobs/{job_id}/video")
def serve_job_video(job_id: str):
    """업로드된 원본 영상 스트리밍."""
    _require_job(job_id)
    metadata = _load_metadata(job_id)
    source = metadata.get("source_filename") or ""
    ext = Path(source).suffix if source else ".mp4"
    video_path = UPLOADS_DIR / f"{job_id}{ext}"
    if not video_path.exists():
        video_path = UPLOADS_DIR / f"{job_id}.mp4"
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="원본 영상 파일을 찾을 수 없습니다.")
    return FileResponse(str(video_path), media_type="video/mp4",
                        filename=Path(source).name if source else f"{job_id}.mp4")


class ManualMarkBody(BaseModel):
    video_sec: float
    event_type: str = "unknown"


@app.post("/api/jobs/{job_id}/marks")
def create_job_mark(job_id: str, body: ManualMarkBody) -> dict[str, Any]:
    """원본 영상 타임스탬프 기준 이벤트 수동 마킹 → event_marks.json 저장."""
    _require_job(job_id)
    metadata = _load_metadata(job_id)
    fps = float(metadata.get("fps") or 30.0)

    event_type = str(body.event_type).strip().lower() or "unknown"
    if event_type not in EVENT_TYPE_SET:
        event_type = "unknown"

    frame = int(round(body.video_sec * fps))
    mark_key = f"{job_id}_manual_{frame}"

    marks = _load_event_marks()
    marks[mark_key] = {
        "mark_key": mark_key,
        "job_id": job_id,
        "fps": fps,
        "video_sec": float(body.video_sec),
        "event_frame_original": frame,
        "event_type": event_type,
        "is_manual": True,
        "marked_at": time.time(),
    }
    _save_event_marks(marks)
    return {"ok": True, "mark_key": mark_key, "mark": marks[mark_key]}


@app.get("/api/jobs/{job_id}/marks")
def list_job_marks(job_id: str) -> dict[str, Any]:
    """해당 job의 수동 마킹 목록."""
    _require_job(job_id)
    marks = _load_event_marks()
    job_marks = {k: v for k, v in marks.items()
                 if v.get("job_id") == job_id and v.get("is_manual")}
    return {"marks": job_marks}


@app.delete("/api/jobs/{job_id}/marks/{mark_key}")
def delete_job_mark(job_id: str, mark_key: str) -> dict[str, Any]:
    """수동 마킹 삭제."""
    _require_job(job_id)
    marks = _load_event_marks()
    if mark_key not in marks or marks[mark_key].get("job_id") != job_id:
        raise HTTPException(status_code=404, detail="마크를 찾을 수 없습니다.")
    del marks[mark_key]
    _save_event_marks(marks)
    return {"ok": True}


@app.get("/api/event_types")
def list_event_types() -> dict[str, Any]:
    """UI 드롭다운에 사용할 표준 이벤트 타입 목록."""
    return {"types": EVENT_TYPES}


@app.get("/api/mode")
def get_mode() -> dict[str, Any]:
    """현재 XGB 활성화 모드 반환 (UI 배지 표시용)."""
    return {"use_xgb": bool(USE_XGB)}


@app.get("/api/bgm")
def list_bgm() -> dict[str, Any]:
    """app/static/bgm/ 폴더의 BGM 파일 목록 반환."""
    tracks = sorted(
        f.name for f in BGM_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}
    )
    return {"tracks": tracks}


@app.post("/api/bgm/upload")
async def upload_bgm(file: UploadFile = File(...)) -> dict[str, Any]:
    """MP3 등 오디오 파일을 BGM 폴더에 업로드."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="지원하지 않는 파일 형식입니다.")
    safe_name = Path(file.filename).name
    dest = BGM_DIR / safe_name
    with dest.open("wb") as f:
        f.write(await file.read())
    return {"name": safe_name}


def _fit_clip(clip, target_size: tuple[int, int]):
    """클립을 target_size 에 맞춰 종횡비를 유지한 채 스케일 + 레터박스(검은 여백)로 채운다.

    같은 종횡비(예: 4K 16:9 → 1080p 16:9)면 여백 없이 꽉 채워지고,
    종횡비가 다르면 위/아래(또는 좌/우)에 검은 여백을 둬 왜곡을 막는다.
    이미 target_size 와 동일하면 원본을 그대로 반환(불필요한 재인코딩 방지).
    """
    tw, th = target_size
    cw, ch = clip.size
    if (cw, ch) == (tw, th):
        return clip
    scale = min(tw / cw, th / ch)
    new_w = max(2, round(cw * scale / 2) * 2)  # 짝수(libx264 요구)
    new_h = max(2, round(ch * scale / 2) * 2)
    try:
        scaled = clip.resized((new_w, new_h))
    except AttributeError:
        scaled = clip.resize((new_w, new_h))
    if (new_w, new_h) == (tw, th):
        return scaled
    from moviepy import ColorClip
    bg = ColorClip(size=(tw, th), color=(0, 0, 0), duration=clip.duration)
    return CompositeVideoClip([bg, scaled.with_position("center")], size=(tw, th))


def _apply_transitions(clips: list, t: float) -> tuple[list, float]:
    """클립 목록에 크로스페이드 트랜지션 적용. (수정된 클립 리스트, padding값) 반환."""
    if t <= 0 or len(clips) <= 1:
        return clips, 0.0
    min_dur = min(c.duration for c in clips)
    t = min(t, min_dur / 2.5)
    result = []
    for i, clip in enumerate(clips):
        effects = []
        if i > 0:
            effects.append(vfx.CrossFadeIn(t))
        if i < len(clips) - 1:
            effects.append(vfx.CrossFadeOut(t))
        result.append(clip.with_effects(effects) if effects else clip)
    return result, -t


BGM_EXTENSIONS = {".mp3", ".wav", ".ogg", ".m4a", ".aac", ".flac"}


def _mix_bgm(merged_clip, bgm_name: str, bgm_volume: float):
    """최종 합쳐진 클립에 BGM을 믹스해 반환. 원본 오디오는 유지."""
    from moviepy import AudioFileClip, CompositeAudioClip

    bgm_path = BGM_DIR / bgm_name
    if not bgm_path.exists() or bgm_path.suffix.lower() not in BGM_EXTENSIONS:
        return merged_clip

    total_dur = merged_clip.duration
    bgm_raw = AudioFileClip(str(bgm_path))

    # BGM이 영상보다 짧으면 루프, 길면 트림
    if bgm_raw.duration < total_dur:
        loops = int(total_dur / bgm_raw.duration) + 1
        from moviepy import concatenate_audioclips
        bgm_raw = concatenate_audioclips([bgm_raw] * loops)
    bgm_clip = bgm_raw.subclipped(0, total_dur)

    # 끝에서 2초 페이드 아웃
    fade_dur = min(2.0, total_dur * 0.1)
    bgm_clip = bgm_clip.with_effects([afx.AudioFadeOut(fade_dur)])
    bgm_clip = bgm_clip.with_volume_scaled(max(0.0, min(bgm_volume, 2.0)))

    if merged_clip.audio is not None:
        mixed = CompositeAudioClip([merged_clip.audio, bgm_clip])
    else:
        mixed = bgm_clip

    return merged_clip.with_audio(mixed)


def _mix_bgm_tracks(merged_clip, tracks: list[BgmTrack]):
    """편집기에서 드래그로 배치한 BGM 트랙들을 합성. 트랙별 fade in/out 지원.

    각 트랙은 (name, start_sec, duration_sec, volume, fade_in_sec, fade_out_sec).
    BGM 파일이 구간보다 짧으면 루프, 길면 잘라낸다.
    """
    from moviepy import AudioFileClip, CompositeAudioClip, concatenate_audioclips

    total_dur = float(merged_clip.duration or 0.0)
    if total_dur <= 0:
        return merged_clip

    pieces: list = []
    if merged_clip.audio is not None:
        pieces.append(merged_clip.audio)

    for t in tracks:
        bgm_path = BGM_DIR / Path(t.name).name
        if not bgm_path.exists() or bgm_path.suffix.lower() not in BGM_EXTENSIONS:
            continue
        start_s = max(0.0, float(t.start_sec))
        if start_s >= total_dur:
            continue
        seg_dur = max(0.5, min(float(t.duration_sec), total_dur - start_s))
        if seg_dur <= 0:
            continue

        raw = AudioFileClip(str(bgm_path))
        if raw.duration < seg_dur:
            loops = int(seg_dur / raw.duration) + 1
            raw = concatenate_audioclips([raw] * loops)
        seg = raw.subclipped(0, seg_dur)

        effects = []
        fade_in = max(0.0, min(float(t.fade_in_sec), seg_dur / 2.0))
        fade_out = max(0.0, min(float(t.fade_out_sec), seg_dur / 2.0))
        if fade_in > 0:
            effects.append(afx.AudioFadeIn(fade_in))
        if fade_out > 0:
            effects.append(afx.AudioFadeOut(fade_out))
        if effects:
            seg = seg.with_effects(effects)
        seg = seg.with_volume_scaled(max(0.0, min(float(t.volume), 2.0)))
        seg = seg.with_start(start_s)
        pieces.append(seg)

    if not pieces:
        return merged_clip

    return merged_clip.with_audio(CompositeAudioClip(pieces))


class ExportOrderBody(BaseModel):
    clip_order: list[str] | None = None
    transition_sec: float = 0.5
    audio_volume: float = 1.0
    bgm_name: str = ""
    bgm_volume: float = 0.3


@app.post("/api/jobs/{job_id}/export")
def export_selected(job_id: str, body: ExportOrderBody | None = Body(default=None)) -> dict[str, Any]:
    job_dir = _require_job(job_id)
    metadata = _load_metadata(job_id)
    selected = metadata.get("selected", {})

    if body and body.clip_order:
        selected_clips = [name for name in body.clip_order if selected.get(name)]
    else:
        selected_clips = [name for name, is_selected in selected.items() if is_selected]
        # 시간순 자동 정렬: clip_timestamps.start 기준, 없으면 파일명 순
        ts = metadata.get("clip_timestamps") or {}
        selected_clips.sort(key=lambda n: ts.get(n, {}).get("start", float("inf")))

    if not selected_clips:
        raise HTTPException(status_code=400, detail="선택된 클립이 없습니다.")

    clips_dir = job_dir / "clips"
    clip_paths = [clips_dir / name for name in selected_clips if (clips_dir / name).exists()]
    if not clip_paths:
        raise HTTPException(status_code=404, detail="선택된 클립 파일을 찾을 수 없습니다.")

    # 트랜지션·BGM은 실사용 모드(USE_XGB=1)에서만 적용
    transition_sec = (body.transition_sec if body and USE_XGB else 0.0)
    audio_volume = max(0.0, min(float(body.audio_volume if body else 1.0), 2.0))
    bgm_name = (body.bgm_name if body and USE_XGB else "")
    bgm_volume = (body.bgm_volume if body else 0.3)
    video_clips = [_fit_clip(VideoFileClip(str(path)), EXPORT_TARGET_SIZE) for path in clip_paths]
    final_clips, padding = _apply_transitions(video_clips, transition_sec)
    merged = concatenate_videoclips(final_clips, padding=padding, method="compose")
    if audio_volume != 1.0 and merged.audio is not None:
        merged = merged.with_volume_scaled(audio_volume)
    if bgm_name:
        merged = _mix_bgm(merged, bgm_name, bgm_volume)
    output_path = job_dir / "merged_selected.mp4"
    merged.write_videofile(str(output_path), codec="libx264", audio_codec="aac", temp_audiofile="temp-merged-audio.m4a", remove_temp=True, preset="veryfast", threads=os.cpu_count() or 4)

    for clip in video_clips:
        clip.close()
    merged.close()

    metadata["merged"] = output_path.name
    _save_metadata(job_id, metadata)
    return {"ok": True, "merged": output_path.name}


@app.post("/api/jobs/{job_id}/image-cards")
async def upload_image_card(job_id: str, file: UploadFile = File(...)) -> dict[str, Any]:
    """이미지 카드 업로드 (타이틀 카드 등)."""
    _require_job(job_id)
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")
    safe_name = Path(file.filename).name
    ext = Path(safe_name).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}:
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다 (jpg/png/gif/bmp/webp).")
    img_dir = _job_dir(job_id) / "image_cards"
    img_dir.mkdir(exist_ok=True, parents=True)
    saved = img_dir / safe_name
    with saved.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"ok": True, "name": safe_name}


@app.post("/api/jobs/{job_id}/external-clips")
async def upload_external_clip(
    job_id: str, file: UploadFile = File(...), register: bool = False
) -> dict[str, Any]:
    """외부 영상 파일을 편집기 클립으로 추가 (분석 없이 타임라인에서 사용).

    runs/{job_id}/clips/ 에 `ext_` 접두사로 저장 → _locate_clip / export_timeline 이 그대로 처리.
    register=False(기본): metadata.clips 에 등록하지 않아 편집기 타임라인에서만 사용.
    register=True: metadata.clips 에 등록 → 하이라이트 후보 그리드에 노출되어 컷 편집 가능.
    """
    _require_job(job_id)
    if not file.filename:
        raise HTTPException(status_code=400, detail="파일이 없습니다.")
    ext = Path(file.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}:
        raise HTTPException(
            status_code=400,
            detail="영상 파일만 업로드 가능합니다 (mp4/mov/avi/mkv/webm/m4v).",
        )
    clips_dir = _job_dir(job_id) / "clips"
    clips_dir.mkdir(exist_ok=True, parents=True)
    safe_stem = re.sub(r"[^\w가-힣.\-]", "_", Path(file.filename).stem).strip("_") or "external"
    name = f"ext_{safe_stem}{ext}"
    saved = clips_dir / name
    i = 1
    while saved.exists():
        name = f"ext_{safe_stem}_{i}{ext}"
        saved = clips_dir / name
        i += 1
    with saved.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    if register:
        metadata = _load_metadata(job_id)
        clips = metadata.setdefault("clips", [])
        if name not in clips:
            clips.append(name)
        # 편집기에서 추가한 외부 영상은 사용 의도가 명확하므로 selected=True 로 등록.
        # → "클립 불러오기"의 selected 필터(server-side meta.selected)를 통과해 다시 불러와짐.
        metadata.setdefault("selected", {})[name] = True
        _save_metadata(job_id, metadata)

    return {"ok": True, "name": name}


@app.get("/api/jobs/{job_id}/image-cards")
def list_image_cards(job_id: str) -> dict[str, Any]:
    """업로드된 이미지 카드 목록."""
    _require_job(job_id)
    img_dir = _job_dir(job_id) / "image_cards"
    if not img_dir.exists():
        return {"cards": []}
    cards = sorted(f.name for f in img_dir.iterdir() if f.is_file())
    return {"cards": cards}


@app.delete("/api/jobs/{job_id}/image-cards/{filename}")
def delete_image_card(job_id: str, filename: str) -> dict[str, Any]:
    """이미지 카드 삭제."""
    _require_job(job_id)
    if "/" in filename or "\\" in filename or filename.startswith(".."):
        raise HTTPException(status_code=400, detail="잘못된 파일명")
    img_path = _job_dir(job_id) / "image_cards" / filename
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="이미지 카드를 찾을 수 없습니다.")
    img_path.unlink()
    return {"ok": True}


class OverlayDef(BaseModel):
    kind: str           # "scoreboard" | "scorer"
    enabled: bool = True
    # ── 이미지 모드 (업로드한 템플릿 사진을 그대로 오버레이) ──
    image_path: str = ""        # runs/{job_id}/overlay_images/ 내 파일명
    image_job_id: str | None = None  # 이미지가 저장된 job (없으면 export job_id 사용)
    width_pct: float = 30.0     # 이미지 가로 (% of video width)
    height_pct: float = 0.0     # 이미지 세로 (% of video height); 0이면 비율 유지
    # ── 그리기 모드 (기존 자동 생성) ──
    home: str = ""
    home_score: int = 0
    away: str = ""
    away_score: int = 0
    number: str = ""    # scorer: 등번호
    name: str = ""      # scorer: 선수명
    bg_color: str = "#FF7400"   # 액센트 블록 배경색 (hex)
    x_pct: float = 25.0     # 박스 좌상단 X (% of width)
    y_pct: float = 5.0      # 박스 좌상단 Y (% of height)
    start_sec: float = 0.0
    duration_sec: float = 5.0


class BgmTrack(BaseModel):
    name: str               # app/static/bgm/ 내 파일명
    start_sec: float = 0.0
    duration_sec: float = 10.0
    volume: float = 0.3
    fade_in_sec: float = 1.0
    fade_out_sec: float = 1.0


class TimelineItem(BaseModel):
    type: str  # 'clip' | 'image'
    name: str
    duration: float = 3.0  # 이미지 카드 표시 시간 (초)
    job_id: str | None = None  # 항목별 출처 job (멀티 job 타임라인 지원)


class TimelineExportBody(BaseModel):
    timeline: list[TimelineItem]
    transition_sec: float = 0.5
    audio_volume: float = 1.0
    bgm_name: str = ""              # legacy: 전체 단일 BGM
    bgm_volume: float = 0.3
    bgm_tracks: list[BgmTrack] = [] # 트랙 기반 다중 BGM (편집기에서 드래그로 배치)
    overlays: list[OverlayDef] = []


def _snap_overlay_timing(start_sec: float, duration_sec: float, clip_duration: float, fps: float) -> tuple[float, float]:
    """오버레이 시작/길이를 프레임 경계에 스냅 → 등장·소멸이 편집 시점과 정확히 일치.

    moviepy CompositeVideoClip 은 프레임 t=i/fps 에서 start<=t<start+dur 인 클립을 표시.
    start_sec 이 프레임 격자에 안 맞으면 ±1프레임 흔들려 "딜레이"로 보일 수 있어 round 로 맞춘다.
    """
    fps = fps if fps and fps > 0 else 30.0
    start_s = max(0.0, round(float(start_sec) * fps) / fps)
    raw_dur = min(float(duration_sec), max(0.0, clip_duration - start_s))
    dur_s = max(1.0 / fps, round(raw_dur * fps) / fps)
    return start_s, dur_s


def _make_overlay_clip(
    ov: OverlayDef,
    video_size: tuple[int, int],
    clip_duration: float,
    fallback_job_id: str | None = None,
    fps: float = 30.0,
):
    """오버레이 클립 생성. 이미지 모드/그리기 모드를 분기."""
    from PIL import Image, ImageDraw, ImageFont

    W, H = video_size

    # ── 이미지 모드: 업로드된 템플릿 사진을 오버레이로 합성 ──
    if ov.image_path:
        img_job_id = ov.image_job_id or fallback_job_id
        if not img_job_id:
            return None
        img_path = _job_dir(img_job_id) / "overlay_images" / Path(ov.image_path).name
        if not img_path.exists():
            print(f"[overlay] 이미지 없음: {img_path}")
            return None
        try:
            pil_img = Image.open(img_path).convert("RGBA")
        except Exception as e:
            print(f"[overlay] 이미지 로드 실패: {e}")
            return None

        target_w = max(50, int(W * max(5.0, min(float(ov.width_pct), 100.0)) / 100.0))
        if ov.height_pct > 0:
            target_h = max(20, int(H * min(float(ov.height_pct), 100.0) / 100.0))
        else:
            ratio = target_w / max(1, pil_img.width)
            target_h = max(20, int(pil_img.height * ratio))
        pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)

        bx = max(0, min(int((ov.x_pct / 100.0) * W), max(0, W - target_w)))
        by = max(0, min(int((ov.y_pct / 100.0) * H), max(0, H - target_h)))

        # ── 향후 확장: 이미지 템플릿 위에 점수/팀명/등번호/선수명을 직접 그리기 ──
        # # 사용자가 편집기에서 텍스트 위치(예: ov.text_anchors)를 지정하면
        # # 아래처럼 ImageDraw로 합성 후 ImageClip 으로 내보낼 수 있음.
        # # 현재는 이미지 그대로 사용. 텍스트 편집 기능은 추후 활성화.
        # from PIL import ImageDraw as _ID
        # _draw = _ID.Draw(pil_img)
        # if ov.kind == "scoreboard":
        #     # ov.text_anchors = {"home": (x, y, size), "score": (...), "away": (...)}
        #     # for key, val in zip(("home","score","away"), (ov.home, f"{ov.home_score}-{ov.away_score}", ov.away)):
        #     #     anchor = ov.text_anchors.get(key)
        #     #     if anchor:
        #     #         tx, ty, ts = anchor
        #     #         _draw.text((tx, ty), val, font=_font(ts), fill=(255,255,255,255))
        #     pass
        # elif ov.kind == "scorer":
        #     # ov.text_anchors = {"number": (...), "name": (...)}
        #     pass

        # 전체 화면 캔버스 대신 '작은 이미지 + 위치'로 합성 → 매 프레임 전체 알파합성 비용 제거
        arr = np.array(pil_img)
        start_s, dur_s = _snap_overlay_timing(ov.start_sec, ov.duration_sec, clip_duration, fps)
        return (ImageClip(arr, duration=dur_s)
                .with_start(start_s).with_fps(fps).with_position((bx, by)))

    # ── 그리기 모드: 기존 자동 생성 (Pillow로 박스/텍스트 직접 그림) ──
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    def _font(size: int):
        for path in (
            "/System/Library/Fonts/AppleSDGothicNeo.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                pass
        return ImageFont.load_default()

    def _hex_to_rgba(h: str, alpha: int = 127) -> tuple[int, int, int, int]:
        s = (h or "").lstrip("#")
        if len(s) == 6:
            try:
                return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), alpha)
            except ValueError:
                pass
        return (255, 116, 0, alpha)

    def _load_logo():
        logo_path = BASE_DIR / "app" / "static" / "overlay_logo.png"
        if logo_path.exists():
            try:
                return Image.open(logo_path).convert("RGBA")
            except Exception:
                pass
        return None

    ACCENT     = _hex_to_rgba(ov.bg_color, alpha=127)   # 약 50% 불투명
    WHITE      = (255, 255, 255, 255)
    LIGHT_DIM  = (255, 255, 255, 215)
    DARK_TEXT  = (20, 14, 8, 255)
    logo_img   = _load_logo()

    def _center_y(box_top: int, box_h: int, bbox, glyph_h: int) -> int:
        # bbox[1]은 폰트의 ascent 오프셋 보정용
        return box_top + (box_h - glyph_h) // 2 - bbox[1]

    if ov.kind == "scoreboard":
        font_team  = _font(max(20, int(H * 0.04)))
        font_score = _font(max(26, int(H * 0.052)))

        home_text  = ov.home or "HOME"
        score_text = f"{ov.home_score} - {ov.away_score}"
        away_text  = ov.away or "AWAY"

        hb = draw.textbbox((0, 0), home_text,  font=font_team)
        sb = draw.textbbox((0, 0), score_text, font=font_score)
        ab = draw.textbbox((0, 0), away_text,  font=font_team)
        hw, hh = hb[2]-hb[0], hb[3]-hb[1]
        sw, sh = sb[2]-sb[0], sb[3]-sb[1]
        aw_, ah = ab[2]-ab[0], ab[3]-ab[1]

        pad_x = max(14, int(W * 0.013))
        pad_y = max(8, int(H * 0.014))
        sep   = max(10, int(W * 0.01))
        block_h = sh + pad_y * 2

        # 로고 크기 = 블록 높이 (사각형 바깥에서 시각적 균형)
        logo_h = block_h
        logo_w = 0
        logo_resized = None
        if logo_img is not None and logo_img.height > 0:
            ratio = logo_img.width / logo_img.height
            logo_w = int(logo_h * ratio)
            logo_resized = logo_img.resize((logo_w, logo_h), Image.LANCZOS)

        # 사각형 너비 (로고 제외)
        rect_w = pad_x + hw + sep + sw + sep + aw_ + pad_x
        # 전체 너비 (로고 + 간격 + 사각형)
        total_w = (logo_w + sep if logo_w else 0) + rect_w

        bx = max(0, min(int((ov.x_pct / 100.0) * W), W - total_w))
        by = max(0, min(int((ov.y_pct / 100.0) * H), H - block_h))

        # 로고는 사각형 왼쪽 바깥에 배치
        rect_x = bx
        if logo_resized is not None:
            img.paste(logo_resized, (bx, by + (block_h - logo_h) // 2), logo_resized)
            rect_x = bx + logo_w + sep

        # 액센트 블록 (로고 제외)
        draw.rectangle([rect_x, by, rect_x + rect_w, by + block_h], fill=ACCENT)

        x = rect_x + pad_x
        draw.text((x, _center_y(by, block_h, hb, hh)), home_text,  font=font_team,  fill=WHITE)
        x += hw + sep
        draw.text((x, _center_y(by, block_h, sb, sh)), score_text, font=font_score, fill=WHITE)
        x += sw + sep
        draw.text((x, _center_y(by, block_h, ab, ah)), away_text,  font=font_team,  fill=WHITE)

    elif ov.kind == "scorer":
        font_num   = _font(max(22, int(H * 0.045)))
        font_label = _font(max(12, int(H * 0.022)))
        font_name  = _font(max(22, int(H * 0.042)))

        number_text = ov.number or "0"
        label_text  = "GOAL!"
        name_text   = ov.name or "PLAYER"

        nb  = draw.textbbox((0, 0), number_text, font=font_num)
        lb  = draw.textbbox((0, 0), label_text,  font=font_label)
        nmb = draw.textbbox((0, 0), name_text,   font=font_name)
        nw_, nh_  = nb[2]-nb[0], nb[3]-nb[1]
        lw_, lh_  = lb[2]-lb[0], lb[3]-lb[1]
        nmw, nmh  = nmb[2]-nmb[0], nmb[3]-nmb[1]

        pad_x = max(12, int(W * 0.011))
        pad_y = max(8, int(H * 0.012))
        sep   = max(10, int(W * 0.01))
        text_inner_h = lh_ + 4 + nmh
        block_h = text_inner_h + pad_y * 2

        # 로고 크기 = 블록 높이 (사각형 바깥에서 시각적 균형)
        logo_h = block_h
        logo_w = 0
        logo_resized = None
        if logo_img is not None and logo_img.height > 0:
            ratio = logo_img.width / logo_img.height
            logo_w = int(logo_h * ratio)
            logo_resized = logo_img.resize((logo_w, logo_h), Image.LANCZOS)

        # 사각형 너비 (로고 제외)
        rect_w = pad_x + nw_ + sep + max(lw_, nmw) + pad_x
        # 전체 너비 (로고 + 간격 + 사각형)
        total_w = (logo_w + sep if logo_w else 0) + rect_w

        bx = max(0, min(int((ov.x_pct / 100.0) * W), W - total_w))
        by = max(0, min(int((ov.y_pct / 100.0) * H), H - block_h))

        # 로고는 사각형 왼쪽 바깥에 배치
        rect_x = bx
        if logo_resized is not None:
            img.paste(logo_resized, (bx, by + (block_h - logo_h) // 2), logo_resized)
            rect_x = bx + logo_w + sep

        # 액센트 블록 (테두리 없음, 로고 제외)
        draw.rectangle([rect_x, by, rect_x + rect_w, by + block_h], fill=ACCENT)

        x = rect_x + pad_x
        # 번호 (큰 글씨, 세로 중앙)
        draw.text((x, _center_y(by, block_h, nb, nh_)), number_text, font=font_num, fill=WHITE)
        x += nw_ + sep

        # GOAL! + 선수명 (2줄)
        text_top = by + (block_h - text_inner_h) // 2
        draw.text((x, text_top - lb[1]), label_text, font=font_label, fill=LIGHT_DIM)
        draw.text((x, text_top + lh_ + 4 - nmb[1]), name_text, font=font_name, fill=WHITE)

    else:
        return None

    # 그려진 영역(불투명 bbox)만 잘라 '작은 이미지 + 위치'로 합성 → 합성 비용 절감
    bbox = img.getbbox()
    if bbox:
        pos = (bbox[0], bbox[1])
        arr = np.array(img.crop(bbox))
    else:
        pos = (0, 0)
        arr = np.array(img)
    start_s, dur_s = _snap_overlay_timing(ov.start_sec, ov.duration_sec, clip_duration, fps)
    return (ImageClip(arr, duration=dur_s)
            .with_start(start_s).with_fps(fps).with_position(pos))


@app.post("/api/jobs/{job_id}/export/timeline")
def export_timeline(job_id: str, body: TimelineExportBody) -> dict[str, Any]:
    """편집기 타임라인 (영상 클립 + 이미지 카드 혼합) → 최종 영상 합치기."""
    job_dir = _require_job(job_id)
    items = body.timeline

    if not items:
        raise HTTPException(status_code=400, detail="타임라인이 비어 있습니다.")

    # 출력 해상도는 EXPORT_TARGET_SIZE 로 고정(해상도 혼합 시 축소-박힘 방지). fps만 첫 클립에서 결정.
    video_size = EXPORT_TARGET_SIZE
    fps = 30.0
    has_clip = False
    for item in items:
        if item.type == "clip":
            item_job = item.job_id or job_id
            clip_path, _ = _locate_clip(item_job, item.name)
            if clip_path is None:
                raise HTTPException(status_code=404, detail=f"클립 없음: {item.name}")
            tmp = VideoFileClip(str(clip_path))
            fps = float(tmp.fps or 30.0)
            tmp.close()
            has_clip = True
            break

    if not has_clip:
        raise HTTPException(status_code=400, detail="영상 클립이 최소 1개 필요합니다.")

    final_clips = []
    try:
        for item in items:
            if item.type == "clip":
                item_job = item.job_id or job_id
                clip_path, _ = _locate_clip(item_job, item.name)
                if clip_path is None:
                    raise HTTPException(status_code=404, detail=f"클립 없음: {item.name}")
                final_clips.append(_fit_clip(VideoFileClip(str(clip_path)), video_size))
            elif item.type == "image":
                item_job = item.job_id or job_id
                item_job_dir = _job_dir(item_job)
                img_path = item_job_dir / "image_cards" / item.name
                if "image_cards" not in str(img_path) or not img_path.resolve().is_relative_to(item_job_dir.resolve()):
                    raise HTTPException(status_code=400, detail="잘못된 이미지 경로")
                if not img_path.exists():
                    raise HTTPException(status_code=404, detail=f"이미지 없음: {item.name}")
                duration = max(0.5, min(float(item.duration), 60.0))
                img_clip = ImageClip(str(img_path), duration=duration)
                try:
                    img_clip = img_clip.resized(video_size)
                except AttributeError:
                    img_clip = img_clip.resize(video_size)
                img_clip = img_clip.with_fps(fps)
                final_clips.append(img_clip)
            else:
                raise HTTPException(status_code=400, detail=f"알 수 없는 타입: {item.type}")

        # 트랜지션·BGM은 실사용 모드(USE_XGB=1)에서만 적용
        t_sec = body.transition_sec if USE_XGB else 0.0
        audio_vol = max(0.0, min(float(body.audio_volume), 2.0))
        transition_clips, padding = _apply_transitions(final_clips, t_sec)
        merged = concatenate_videoclips(transition_clips, padding=padding, method="compose")
        if audio_vol != 1.0 and merged.audio is not None:
            merged = merged.with_volume_scaled(audio_vol)
        if USE_XGB:
            if body.bgm_tracks:
                merged = _mix_bgm_tracks(merged, body.bgm_tracks)
            elif body.bgm_name:
                merged = _mix_bgm(merged, body.bgm_name, body.bgm_volume)

        # 합쳐진 영상에 오버레이 트랙 적용
        if body.overlays:
            ov_clips = []
            for ov in body.overlays:
                if not ov.enabled:
                    continue
                try:
                    oc = _make_overlay_clip(ov, merged.size, merged.duration, fallback_job_id=job_id, fps=fps)
                    if oc is not None:
                        ov_clips.append(oc)
                except Exception as e:
                    print(f"[overlay] 렌더 실패: {e}")
            if ov_clips:
                merged = CompositeVideoClip([merged] + ov_clips)

        output_path = job_dir / "merged_timeline.mp4"
        merged.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-timeline-audio.m4a",
            remove_temp=True,
            preset="veryfast",
            threads=os.cpu_count() or 4,
        )
        merged.close()
    finally:
        for c in final_clips:
            try:
                c.close()
            except Exception:
                pass

    metadata = _load_metadata(job_id)
    metadata["merged_timeline"] = output_path.name
    _save_metadata(job_id, metadata)
    return {"ok": True, "merged": output_path.name}


@app.post("/api/jobs/{job_id}/export/selected")
def export_selected_clips(job_id: str) -> dict[str, Any]:
    job_dir = _require_job(job_id)
    metadata = _load_metadata(job_id)
    selected = metadata.get("selected", {})
    selected_clips = [name for name, is_selected in selected.items() if is_selected]

    if not selected_clips:
        raise HTTPException(status_code=400, detail="선택된 클립이 없습니다.")

    export_dir = EXPORTS_DIR / job_id
    export_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = job_dir / "clips"

    exported = []
    for name in selected_clips:
        src = clips_dir / name
        if not src.exists():
            continue
        dest = export_dir / name
        shutil.copy2(src, dest)
        exported.append(name)

    metadata["exported_dir"] = str(export_dir)
    metadata["exported"] = exported
    _save_metadata(job_id, metadata)
    return {"ok": True, "exported": exported, "dir": str(export_dir)}

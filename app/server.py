from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from fastapi import BackgroundTasks, Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from moviepy import VideoFileClip, ImageClip, concatenate_videoclips, vfx
from pydantic import BaseModel

from extract import run_highlight_pipeline, USE_XGB
from log_extract import (
    run_log_pipeline,
    CLIP_BEFORE as LOG_CLIP_BEFORE,
    CLIP_AFTER as LOG_CLIP_AFTER,
    CLIP_MINUTE_SPAN as LOG_CLIP_MINUTE_SPAN,
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
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, cls=_NumpyEncoder), encoding="utf-8")


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
            start_sec = max(0.0, video_sec - LOG_CLIP_MINUTE_SPAN - LOG_CLIP_BEFORE)
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
    saved_video = UPLOADS_DIR / f"{job_id}{ext}"
    with saved_video.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

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


@app.post("/api/log-jobs")
async def create_log_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    log_data: str = Form(...),
    second_half_start_min: int = Form(25),
    second_half_start_sec_extra: int = Form(0),
    highlight_count: int = Form(DEFAULT_HIGHLIGHT_COUNT),
) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="업로드할 파일이 없습니다.")

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
    saved_video = UPLOADS_DIR / f"{job_id}{ext}"
    with saved_video.open("wb") as buffer:
        import shutil as _shutil
        _shutil.copyfileobj(file.file, buffer)

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
    job_dir = _require_job(job_id)
    clip_path, _ = _locate_clip(job_id, clip_name)
    if clip_path is None:
        raise HTTPException(status_code=404, detail="클립을 찾을 수 없습니다.")

    if start < 0 or end <= 0 or end <= start:
        raise HTTPException(status_code=400, detail="끝 시간이 시작 시간보다 커야 합니다.")

    # trimmed 파일은 원래 위치 옆에 저장 (Good/Bad 에서 잘랐으면 그 폴더에 생성)
    trimmed_path = clip_path.with_name(f"trimmed_{clip_name}")
    clip = VideoFileClip(str(clip_path))
    end_time = min(end, clip.duration)
    if start >= end_time:
        clip.close()
        raise HTTPException(status_code=400, detail="잘못된 시간 범위입니다.")

    try:
        trimmed = clip.subclip(start, end_time)
    except AttributeError:
        trimmed = clip.subclipped(start, end_time)
    trimmed.write_videofile(
        str(trimmed_path),
        codec="libx264",
        audio_codec="aac",
        temp_audiofile="temp-trimmed-audio.m4a",
        remove_temp=True,
    )
    trimmed.close()
    clip.close()

    metadata = _load_metadata(job_id)
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
    bgm_clip = bgm_clip.with_effects([vfx.AudioFadeOut(fade_dur)])
    bgm_clip = bgm_clip.with_volume_scaled(max(0.0, min(bgm_volume, 2.0)))

    if merged_clip.audio is not None:
        mixed = CompositeAudioClip([merged_clip.audio, bgm_clip])
    else:
        mixed = bgm_clip

    return merged_clip.with_audio(mixed)


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
    video_clips = [VideoFileClip(str(path)) for path in clip_paths]
    final_clips, padding = _apply_transitions(video_clips, transition_sec)
    merged = concatenate_videoclips(final_clips, padding=padding, method="compose")
    if audio_volume != 1.0 and merged.audio is not None:
        merged = merged.with_volume_scaled(audio_volume)
    if bgm_name:
        merged = _mix_bgm(merged, bgm_name, bgm_volume)
    output_path = job_dir / "merged_selected.mp4"
    merged.write_videofile(str(output_path), codec="libx264", audio_codec="aac", temp_audiofile="temp-merged-audio.m4a", remove_temp=True)

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


class TimelineItem(BaseModel):
    type: str  # 'clip' | 'image'
    name: str
    duration: float = 3.0  # 이미지 카드 표시 시간 (초)


class TimelineExportBody(BaseModel):
    timeline: list[TimelineItem]
    transition_sec: float = 0.5
    audio_volume: float = 1.0
    bgm_name: str = ""
    bgm_volume: float = 0.3


@app.post("/api/jobs/{job_id}/export/timeline")
def export_timeline(job_id: str, body: TimelineExportBody) -> dict[str, Any]:
    """편집기 타임라인 (영상 클립 + 이미지 카드 혼합) → 최종 영상 합치기."""
    job_dir = _require_job(job_id)
    items = body.timeline

    if not items:
        raise HTTPException(status_code=400, detail="타임라인이 비어 있습니다.")

    # 첫 번째 영상 클립에서 해상도·fps 결정
    video_size = None
    fps = 30.0
    for item in items:
        if item.type == "clip":
            clip_path, _ = _locate_clip(job_id, item.name)
            if clip_path is None:
                raise HTTPException(status_code=404, detail=f"클립 없음: {item.name}")
            tmp = VideoFileClip(str(clip_path))
            video_size = tmp.size
            fps = float(tmp.fps or 30.0)
            tmp.close()
            break

    if video_size is None:
        raise HTTPException(status_code=400, detail="영상 클립이 최소 1개 필요합니다.")

    final_clips = []
    try:
        for item in items:
            if item.type == "clip":
                clip_path, _ = _locate_clip(job_id, item.name)
                if clip_path is None:
                    raise HTTPException(status_code=404, detail=f"클립 없음: {item.name}")
                final_clips.append(VideoFileClip(str(clip_path)))
            elif item.type == "image":
                img_path = job_dir / "image_cards" / item.name
                if "image_cards" not in str(img_path) or not img_path.resolve().is_relative_to(job_dir.resolve()):
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
        if body.bgm_name and USE_XGB:
            merged = _mix_bgm(merged, body.bgm_name, body.bgm_volume)
        output_path = job_dir / "merged_timeline.mp4"
        merged.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile="temp-timeline-audio.m4a",
            remove_temp=True,
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

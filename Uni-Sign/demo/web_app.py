import argparse
import base64
import cgi
import glob
import hashlib
import html
import json
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, quote, urlparse
from uuid import uuid4

import cv2
import numpy as np
from rtmlib import Wholebody, draw_skeleton

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def _find_ffmpeg_executable() -> str:
    if imageio_ffmpeg is not None:
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            if ffmpeg_exe and os.path.isfile(ffmpeg_exe) and os.access(ffmpeg_exe, os.X_OK):
                return ffmpeg_exe
        except Exception:
            pass

    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg

    search_patterns = [
        os.path.join(Path.home().as_posix(), ".local", "lib", "python*", "site-packages", "imageio_ffmpeg", "binaries", "ffmpeg-*"),
        "/home/*/.local/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*",
        "/usr/local/lib/python*/site-packages/imageio_ffmpeg/binaries/ffmpeg-*",
    ]
    for pattern in search_patterns:
        for candidate in sorted(glob.glob(pattern), reverse=True):
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate

    return ""

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Allow running this script from any working directory.
_UNI_SIGN_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _UNI_SIGN_ROOT not in sys.path:
    sys.path.insert(0, _UNI_SIGN_ROOT)

import utils as utils
from online_inference import get_runtime_device, load_model, predict_video, load_pose_file


DEFAULT_CHECKPOINT = os.path.join(
    _UNI_SIGN_ROOT,
    "runs",
    "a10_stage3_poseonly_cecsl_20260508_1351",
    "best_checkpoint.pth",
)
DEFAULT_SAMPLE_VIDEO_DIR = os.path.join(
    _UNI_SIGN_ROOT,
    "dataset",
    "CE-CSL",
    "CE-CSL",
    "video",
    "test",
    "A",
)
DEFAULT_SAMPLE_POSE_DIR = os.path.join(
    _UNI_SIGN_ROOT,
    "dataset",
    "CE-CSL",
    "pose_format",
    "test",
    "A",
)
DEFAULT_SAMPLE_SPECS = [(f"test-{index:05d}", f"test-{index:04d}") for index in range(1, 11)]
DEFAULT_CE_CSL_TRAINING_SUMMARY = {
    # Measured from Uni-Sign/dataset/CE-CSL/CE-CSL/label/train.csv on 2026-05-17.
    # The full train split contains 4,973 clips with average duration 6.304 s.
    "avg_duration_seconds": 6.304,
    "avg_fps": 29.663,
    "avg_frame_count": 187,
}
WEBSOCKET_ACCEPT_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
WHOLEBODY_HAND_KEYPOINT_INDICES = tuple(range(91, 133))
WHOLEBODY_UPPER_BODY_KEYPOINT_INDICES = tuple(range(0, 13)) + WHOLEBODY_HAND_KEYPOINT_INDICES


class UploadTooLargeError(ValueError):
    pass


class SampleVideoManager:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)
        self.pose_dir = Path(DEFAULT_SAMPLE_POSE_DIR)
        self.samples = []
        for sample_id, display_id in DEFAULT_SAMPLE_SPECS:
            file_path = self.root_dir / f"{sample_id}.mp4"
            pose_path = self.pose_dir / f"{sample_id}.pkl"
            if not file_path.exists() or not file_path.is_file() or not pose_path.exists() or not pose_path.is_file():
                continue
            fps = 0.0
            frame_count = 0
            duration_seconds = 0.0
            cap = cv2.VideoCapture(str(file_path))
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if fps > 0 and frame_count > 0:
                    duration_seconds = frame_count / fps
            cap.release()
            self.samples.append(
                {
                    "id": sample_id,
                    "label": display_id,
                    "filename": f"{display_id}.mp4",
                    "preview_url": f"/sample-media/{sample_id}.mp4",
                    "path": str(file_path),
                    "pose_path": str(pose_path),
                    "fps": fps,
                    "frame_count": frame_count,
                    "duration_seconds": duration_seconds,
                }
            )

    def list(self):
        return [
            {
                "id": sample["id"],
                "label": sample["label"],
                "filename": sample["filename"],
                "preview_url": sample["preview_url"],
            }
            for sample in self.samples
        ]

    def get(self, sample_id: str):
        sample_id = str(sample_id or "").strip()
        for sample in self.samples:
            if sample["id"] == sample_id:
                return dict(sample)
        return None

    def resolve_media(self, media_name: str) -> Optional[Path]:
        safe_name = os.path.basename(media_name)
        for sample in self.samples:
            expected_name = f"{sample['id']}.mp4"
            if safe_name == expected_name:
                return Path(sample["path"])
        return None

    def cache_key(self, sample: dict) -> str:
        return f"sample-pose-{sample['id']}.mp4"

    def summary(self) -> dict:
        if not self.samples:
            return {
                "avg_duration_seconds": 8.0,
                "avg_fps": 25.0,
                "avg_frame_count": 200,
            }

        durations = [sample["duration_seconds"] for sample in self.samples if sample.get("duration_seconds")]
        fps_values = [sample["fps"] for sample in self.samples if sample.get("fps")]
        frame_counts = [sample["frame_count"] for sample in self.samples if sample.get("frame_count")]
        return {
            "avg_duration_seconds": sum(durations) / len(durations) if durations else 8.0,
            "avg_fps": sum(fps_values) / len(fps_values) if fps_values else 25.0,
            "avg_frame_count": int(sum(frame_counts) / len(frame_counts)) if frame_counts else 200,
        }


class UploadManager:
    def __init__(self, max_upload_bytes: int, max_chunk_bytes: int):
        self.max_upload_bytes = max_upload_bytes
        self.max_chunk_bytes = max_chunk_bytes
        self.root_dir = Path(tempfile.gettempdir()) / "uni_sign_web_uploads"
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.sessions = {}

    def start(self, filename: str, total_size: int) -> str:
        if total_size <= 0:
            raise ValueError("上传文件为空，请重新选择视频。")
        if total_size > self.max_upload_bytes:
            raise UploadTooLargeError(
                f"文件过大：{total_size / (1024 * 1024):.1f} MB，当前服务上限为 {self.max_upload_bytes / (1024 * 1024):.1f} MB。"
            )

        upload_id = uuid4().hex
        session_dir = self.root_dir / upload_id
        session_dir.mkdir(parents=True, exist_ok=True)
        target_path = session_dir / f"upload{Path(filename).suffix or '.mp4'}"

        with self.lock:
            self.sessions[upload_id] = {
                "filename": filename,
                "total_size": total_size,
                "received": 0,
                "next_index": 0,
                "path": str(target_path),
            }

        return upload_id

    def append_chunk(self, upload_id: str, index: int, chunk: bytes) -> dict:
        if len(chunk) > self.max_chunk_bytes:
            raise UploadTooLargeError(
                f"单个分片过大：{len(chunk) / (1024 * 1024):.1f} MB，当前分片上限为 {self.max_chunk_bytes / (1024 * 1024):.1f} MB。"
            )

        with self.lock:
            session = self.sessions.get(upload_id)
            if session is None:
                raise ValueError("上传会话不存在，请重新选择文件上传。")
            if index != session["next_index"]:
                raise ValueError("上传分片顺序错误，请重新上传。")

            next_received = session["received"] + len(chunk)
            if next_received > session["total_size"]:
                raise ValueError("上传数据超过声明的文件大小，请重新上传。")

            with open(session["path"], "ab") as f:
                f.write(chunk)

            session["received"] = next_received
            session["next_index"] += 1
            return {
                "received": session["received"],
                "total_size": session["total_size"],
            }

    def finish(self, upload_id: str) -> tuple[str, str]:
        with self.lock:
            session = self.sessions.get(upload_id)
            if session is None:
                raise ValueError("上传会话不存在，请重新选择文件上传。")
            if session["received"] != session["total_size"]:
                raise ValueError("文件尚未上传完成，请稍后重试。")
            return session["path"], session["filename"]

    def cleanup(self, upload_id: str):
        with self.lock:
            session = self.sessions.pop(upload_id, None)

        if not session:
            return

        session_dir = Path(session["path"]).parent
        try:
            shutil.rmtree(session_dir)
        except FileNotFoundError:
            pass


class ResultManager:
    def __init__(self):
        self.root_dir = Path(tempfile.gettempdir()) / "uni_sign_web_results"
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def register_pose_video(self, source_path: str) -> str:
        video_id = f"{uuid4().hex}.mp4"
        target_path = self.root_dir / video_id
        shutil.copyfile(source_path, target_path)
        return video_id

    def resolve(self, media_name: str) -> Optional[Path]:
        safe_name = os.path.basename(media_name)
        if not safe_name.endswith(".mp4"):
            return None
        target = self.root_dir / safe_name
        if not target.exists() or not target.is_file():
            return None
        return target


class InferenceJobManager:
    def __init__(self, runner, cleanup_upload, sample_runner=None):
        self.runner = runner
        self.cleanup_upload = cleanup_upload
        self.sample_runner = sample_runner
        self.lock = threading.Lock()
        self.jobs = {}
        self.latest_job_id = ""

    def start(self, upload_id: str, video_path: str, filename: str) -> str:
        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "upload_id": upload_id,
            "filename": filename,
            "status": "queued",
            "message": "任务已创建，等待开始处理。",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "result": "",
            "pose_video_url": "",
            "inference_seconds": None,
            "error": "",
        }
        with self.lock:
            self.jobs[job_id] = job
            self.latest_job_id = job_id

        worker = threading.Thread(
            target=self._run_job,
            args=(job_id, upload_id, video_path),
            daemon=True,
        )
        worker.start()
        return job_id

    def start_sample(self, sample: dict) -> str:
        if self.sample_runner is None:
            raise RuntimeError("默认样本推理未启用。")

        job_id = uuid4().hex
        job = {
            "job_id": job_id,
            "upload_id": "",
            "filename": sample.get("filename", ""),
            "status": "queued",
            "message": "默认样本任务已创建，等待开始处理。",
            "created_at": time.time(),
            "started_at": None,
            "finished_at": None,
            "result": "",
            "pose_video_url": sample.get("preview_url", ""),
            "inference_seconds": None,
            "error": "",
        }
        with self.lock:
            self.jobs[job_id] = job
            self.latest_job_id = job_id

        worker = threading.Thread(
            target=self._run_sample_job,
            args=(job_id, sample),
            daemon=True,
        )
        worker.start()
        return job_id

    def _run_job(self, job_id: str, upload_id: str, video_path: str):
        self._update(job_id, status="running", message="正在生成 pose 视频并执行推理...", started_at=time.time())
        try:
            output = self.runner(video_path)
            self._update(
                job_id,
                status="completed",
                message="推理完成。",
                finished_at=time.time(),
                result=output.get("result", ""),
                pose_video_url=output.get("pose_video_url", ""),
                inference_seconds=output.get("inference_seconds"),
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                message="推理失败。",
                finished_at=time.time(),
                error=str(exc),
            )
        finally:
            self.cleanup_upload(upload_id)

    def _run_sample_job(self, job_id: str, sample: dict):
        self._update(job_id, status="running", message="正在执行默认样本推理...", started_at=time.time())
        try:
            output = self.sample_runner(sample)
            self._update(
                job_id,
                status="completed",
                message="默认样本推理完成。",
                finished_at=time.time(),
                result=output.get("result", ""),
                pose_video_url=output.get("pose_video_url", sample.get("preview_url", "")),
                inference_seconds=output.get("inference_seconds"),
            )
        except Exception as exc:
            self._update(
                job_id,
                status="failed",
                message="默认样本推理失败。",
                finished_at=time.time(),
                error=str(exc),
            )

    def _update(self, job_id: str, **fields):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return
            job.update(fields)

    def get(self, job_id: str):
        with self.lock:
            job = self.jobs.get(job_id)
            if not job:
                return None
            return dict(job)

    def get_by_upload_id(self, upload_id: str):
        with self.lock:
            for job in self.jobs.values():
                if job.get("upload_id") == upload_id:
                    return dict(job)
        return None

    def get_latest(self):
        with self.lock:
            if not self.latest_job_id:
                return None
            job = self.jobs.get(self.latest_job_id)
            if not job:
                return None
            return dict(job)


def render_page(result: str = "", error: str = "", filename: str = "", pose_video_url: str = "", app_args=None, sample_videos=None, realtime_config=None) -> str:
    result_html = ""
    if result:
        result_html = (
            "<section class='card success'>"
            "<h2>推理结果</h2>"
            f"<p class='meta'>视频文件：{html.escape(filename)}</p>"
            f"<pre>{html.escape(result)}</pre>"
            "</section>"
        )

    preview_html = ""
    if pose_video_url:
        preview_html = (
            "<section class='card'>"
            "<h2>Pose 可视化视频</h2>"
            f"<p class='meta'>视频文件：{html.escape(filename)}</p>"
            f"<video controls playsinline preload='metadata' style='width: 100%; border-radius: 12px; background: #000;'>"
            f"<source src='{html.escape(pose_video_url, quote=True)}' type='video/mp4' />"
            "当前浏览器无法播放该视频。"
            "</video>"
            "</section>"
        )

    error_html = ""
    if error:
        error_html = (
            "<section class='card error'>"
            "<h2>处理失败</h2>"
            f"<pre>{html.escape(error)}</pre>"
            "</section>"
        )

    mode_text = "RGB + Pose" if getattr(app_args, "rgb_support", False) else "Pose Only"
    checkpoint_text = getattr(app_args, "finetune", "") or DEFAULT_CHECKPOINT
    dataset_text = getattr(app_args, "dataset", "CSL_Daily")
    device_text = get_runtime_device(app_args) if app_args is not None else "auto"
    max_upload_mb = getattr(app_args, "max_upload_size_mb", 0)
    chunk_mb = getattr(app_args, "chunk_size_mb", 0)
    sample_videos = sample_videos or []
    realtime_config = realtime_config or {}
    sample_options_html = "".join(
        f"<option value='{html.escape(sample['id'], quote=True)}'>{html.escape(sample['filename'])}</option>"
        for sample in sample_videos
    )
    sample_section_html = ""
    if sample_videos:
        first_sample = sample_videos[0]
        sample_section_html = (
            "<section class='card'>"
            "<h2>默认样本体验</h2>"
            "<p>如果暂时不上传文件，可以直接选择 CE-CSL 的默认样本查看效果。</p>"
            "<div class='sample-toolbar'>"
            "<label class='sample-label' for='sampleSelect'>选择样本</label>"
            f"<select id='sampleSelect'>{sample_options_html}</select>"
            "<button id='sampleRunButton' type='button'>使用该样本推理</button>"
            "</div>"
            f"<p id='sampleMeta' class='meta'>当前样本：{html.escape(first_sample['filename'])}</p>"
            "<video id='samplePreview' controls playsinline preload='metadata' style='width: 100%; border-radius: 12px; background: #000;'>"
            f"<source id='samplePreviewSource' src='{html.escape(first_sample['preview_url'], quote=True)}' type='video/mp4' />"
            "当前浏览器无法播放该样本视频。"
            "</video>"
            "</section>"
        )
    status_text = "等待上传。"
    if error:
        status_text = f"处理失败：{error}"
    elif result:
        status_text = f"推理完成：{filename}"
    status_section_html = (
        "<section class='card status'>"
        "<h2>当前状态</h2>"
        f"<pre id='statusBox'>{html.escape(status_text)}</pre>"
        "</section>"
    )
    realtime_section_html = (
        "<section id='realtimeSection' class='card'>"
        "<h2>实时摄像头推理</h2>"
        "<div class='camera-actions'>"
        "<button id='startCameraButton' type='button'>打开摄像头</button>"
        "<button id='stopCameraButton' type='button'>停止摄像头</button>"
        "</div>"
        "<p id='realtimeHint' class='meta'>点击“打开摄像头”后开始实时跟踪；pose 会按最高可用吞吐持续刷新，字幕结果在后台按滑窗异步更新。</p>"
        "<div class='camera-stack'>"
        "<canvas id='cameraPreviewCanvas'></canvas>"
        "<video id='cameraVideo' autoplay muted playsinline aria-hidden='true'></video>"
        "<img id='cameraOverlay' alt='' hidden />"
        "</div>"
        "<canvas id='cameraCaptureCanvas' style='display:none;'></canvas>"
        "<canvas id='cameraOverlayCaptureCanvas' style='display:none;'></canvas>"
        "<section class='card status' style='margin-top: 16px; margin-bottom: 0;'>"
        "<h2 style='font-size:18px;'>后台字幕推理</h2>"
        "<pre id='realtimeSubtitle'>等待摄像头启动。</pre>"
        "</section>"
        "</section>"
    )

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Uni-Sign 视频翻译</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f7fb; color: #1f2937; margin: 0; }}
    .wrap {{ max-width: 860px; margin: 40px auto; padding: 0 20px; }}
    .card {{ background: #fff; border-radius: 16px; padding: 24px; box-shadow: 0 10px 30px rgba(15, 23, 42, 0.08); margin-bottom: 20px; }}
    h1 {{ margin: 0 0 12px; font-size: 32px; }}
    h2 {{ margin-top: 0; font-size: 22px; }}
    p {{ line-height: 1.6; }}
    .meta {{ color: #6b7280; font-size: 14px; }}
    ul {{ padding-left: 18px; color: #4b5563; }}
    input[type=file] {{ display: block; width: 100%; margin: 16px 0; font-size: 15px; }}
    select {{ width: 100%; min-width: 240px; padding: 12px 14px; border: 1px solid #d1d5db; border-radius: 10px; font-size: 15px; background: #fff; }}
    button {{ background: #2563eb; color: #fff; border: 0; border-radius: 10px; padding: 12px 18px; font-size: 16px; cursor: pointer; }}
    button:hover {{ background: #1d4ed8; }}
    .success {{ border-left: 5px solid #16a34a; }}
    .error {{ border-left: 5px solid #dc2626; }}
    .status {{ border-left: 5px solid #2563eb; }}
    pre {{ white-space: pre-wrap; word-break: break-word; background: #f8fafc; border-radius: 10px; padding: 16px; margin: 0; }}
    .sample-toolbar {{ display: flex; gap: 12px; align-items: end; flex-wrap: wrap; margin: 16px 0; }}
    .sample-label {{ display: block; color: #374151; font-size: 14px; margin-bottom: 6px; }}
    .camera-actions {{ display: flex; gap: 12px; flex-wrap: wrap; margin: 16px 0; }}
    .camera-stack {{ position: relative; width: 100%; max-width: 420px; margin: 0 auto; aspect-ratio: 4 / 3; background: #000; border-radius: 12px; overflow: hidden; }}
    .camera-stack canvas, .camera-stack video, .camera-stack img {{ position: absolute; inset: 0; width: 100%; height: 100%; object-fit: cover; }}
    .camera-stack video {{ opacity: 0; pointer-events: none; }}
    .camera-stack img {{ pointer-events: none; }}
    .camera-stack canvas {{ display: block; }}
    .camera-stack img[hidden] {{ display: none; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="card">
      <h1>Uni-Sign 视频翻译</h1>
      <p>上传一段手语视频，服务端会调用 Uni-Sign 在线推理，并返回中文翻译结果。</p>
      <ul>
        <li>当前模式：{html.escape(mode_text)}</li>
        <li>运行设备：{html.escape(device_text)}</li>
        <li>数据集提示词：{html.escape(dataset_text)}</li>
        <li>Checkpoint：{html.escape(checkpoint_text)}</li>
        <li>上传上限：{max_upload_mb} MB（浏览器会自动按 {chunk_mb} MB 分片上传）</li>
      </ul>
      <form id="uploadForm" method="post" action="/predict" enctype="multipart/form-data">
        <input id="videoInput" type="file" name="video" accept="video/*" required />
        <button type="submit">上传并推理</button>
      </form>
    </section>
    {status_section_html}
    {sample_section_html}
    <div id="asyncResultContainer">
      {preview_html}
      {error_html}
      {result_html}
    </div>
    {realtime_section_html}
    <div id="realtimeResultContainer"></div>
  </div>
  <script>
    const uploadForm = document.getElementById('uploadForm');
    const videoInput = document.getElementById('videoInput');
    const statusBox = document.getElementById('statusBox');
    const sampleSelect = document.getElementById('sampleSelect');
    const sampleRunButton = document.getElementById('sampleRunButton');
    const samplePreview = document.getElementById('samplePreview');
    const samplePreviewSource = document.getElementById('samplePreviewSource');
    const sampleMeta = document.getElementById('sampleMeta');
    const asyncResultContainer = document.getElementById('asyncResultContainer');
    const realtimeResultContainer = document.getElementById('realtimeResultContainer');
    const startCameraButton = document.getElementById('startCameraButton');
    const stopCameraButton = document.getElementById('stopCameraButton');
    const cameraPreviewCanvas = document.getElementById('cameraPreviewCanvas');
    const cameraVideo = document.getElementById('cameraVideo');
    const cameraOverlay = document.getElementById('cameraOverlay');
    const cameraCaptureCanvas = document.getElementById('cameraCaptureCanvas');
    const cameraOverlayCaptureCanvas = document.getElementById('cameraOverlayCaptureCanvas');
    const realtimeSubtitle = document.getElementById('realtimeSubtitle');
    const realtimeHint = document.getElementById('realtimeHint');
    const chunkSize = {int(getattr(app_args, 'chunk_size_mb', 4) * 1024 * 1024)};
    const defaultSamples = {json.dumps(sample_videos, ensure_ascii=False)};
    const realtimeConfig = {json.dumps(realtime_config, ensure_ascii=False)};
    let cameraStream = null;
    let captureTimer = null;
    let inferTimer = null;
    let realtimeFrames = [];
    let totalCapturedFrames = 0;
    let realtimeBusy = false;
    let lastRealtimeInferenceAt = 0;
    let latestCapturedFrame = '';
    let realtimeOverlayBusy = false;
    let lastRealtimeOverlayAt = 0;
    let overlayPumpTimer = null;
    let cameraRenderFrameId = 0;
    let realtimeInferenceStartedAt = 0;
    let realtimeStatusTimer = null;
    let realtimeIssueMessage = '';
    let realtimeOverlayRequestSeq = 0;
    let realtimeOverlayAppliedSeq = 0;
    let compositeRecorder = null;
    let compositeRecorderStream = null;
    let compositeRecorderMimeType = '';
    let compositeChunks = [];
    let currentRealtimeClipUrl = '';
    let currentRealtimeOverlayUrl = '';
    let realtimeOverlaySocket = null;
    let realtimeOverlaySocketPromise = null;
    let realtimeOverlaySocketPending = null;
    let realtimeOverlayTransport = 'websocket';
    let realtimeOverlayWsDisabledReason = '';

    function setStatus(message) {{
      statusBox.textContent = message;
    }}

    function escapeHtml(text) {{
      return String(text)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');
    }}

    async function readApiResponse(resp) {{
      const text = await resp.text();
      try {{
        return JSON.parse(text || '{{}}');
      }} catch (error) {{
        const snippet = text.replace(/\s+/g, ' ').trim().slice(0, 200);
        const message = snippet || `HTTP ${{resp.status}}`;
        throw new Error(`接口返回了非 JSON 响应（HTTP ${{resp.status}}）：${{message}}`);
      }}
    }}

    function ensureResultCard(title, filename, content, className) {{
      let card = document.getElementById('dynamicResultCard');
      if (!card) {{
        card = document.createElement('section');
        card.id = 'dynamicResultCard';
        card.className = 'card';
        (asyncResultContainer || document.querySelector('.wrap')).appendChild(card);
      }}
      card.className = `card ${{className}}`;
      card.innerHTML = `<h2>${{escapeHtml(title)}}</h2><p class="meta">视频文件：${{escapeHtml(filename)}}</p><pre>${{escapeHtml(content)}}</pre>`;
    }}

    function ensureVideoCard(filename, videoUrl) {{
      let card = document.getElementById('poseVideoCard');
      if (!card) {{
        card = document.createElement('section');
        card.id = 'poseVideoCard';
        card.className = 'card';
        (asyncResultContainer || document.querySelector('.wrap')).appendChild(card);
      }}
      card.innerHTML = `<h2>Pose 可视化视频</h2><p class="meta">视频文件：${{escapeHtml(filename)}}</p><video controls playsinline preload="metadata" style="width: 100%; border-radius: 12px; background: #000;"><source src="${{escapeHtml(videoUrl)}}" type="video/mp4">当前浏览器无法播放该视频。</video>`;
    }}

    function ensureRealtimeClipCard(resultText, videoUrl) {{
      if (!videoUrl) {{
        return;
      }}
      let card = document.getElementById('realtimeClipCard');
      if (!card) {{
        card = document.createElement('section');
        card.id = 'realtimeClipCard';
        card.className = 'card success';
        (realtimeResultContainer || document.querySelector('.wrap')).appendChild(card);
      }}
      if (currentRealtimeClipUrl && currentRealtimeClipUrl !== videoUrl) {{
        URL.revokeObjectURL(currentRealtimeClipUrl);
      }}
      currentRealtimeClipUrl = videoUrl;
      card.className = 'card success';
      card.innerHTML = `<h2>实时推理录屏</h2><p class="meta">最近一轮实时推理对应的带 pose 录屏结果。</p><p class="meta">字幕：${{escapeHtml(resultText || '当前未识别到稳定字幕。')}}</p><video id="realtimeClipPlayer" controls playsinline preload="metadata" style="width: 100%; border-radius: 12px; background: #000;" src="${{escapeHtml(videoUrl)}}">当前浏览器无法播放该视频。</video>`;
      const player = document.getElementById('realtimeClipPlayer');
      if (player && player.load) {{
        player.load();
      }}
    }}

    function setRealtimeHint(message) {{
      if (realtimeHint) {{
        realtimeHint.textContent = message;
      }}
    }}

    function setRealtimeIssue(message = '') {{
      realtimeIssueMessage = String(message || '').trim();
    }}

    function setRealtimeSubtitle(message) {{
      if (realtimeSubtitle) {{
        realtimeSubtitle.textContent = message;
      }}
    }}

    function setRealtimeOverlay(imageUrl) {{
      if (!cameraOverlay) {{
        return;
      }}
      if (currentRealtimeOverlayUrl && currentRealtimeOverlayUrl !== imageUrl) {{
        URL.revokeObjectURL(currentRealtimeOverlayUrl);
        currentRealtimeOverlayUrl = '';
      }}
      if (imageUrl) {{
        cameraOverlay.hidden = false;
        cameraOverlay.src = imageUrl;
        if (imageUrl.startsWith('blob:')) {{
          currentRealtimeOverlayUrl = imageUrl;
        }}
        return;
      }}
      cameraOverlay.hidden = true;
      cameraOverlay.removeAttribute('src');
    }}

    function getRealtimeWindowMs() {{
      return Math.max(
        Math.round((realtimeConfig.window_seconds || 8) * 1000),
        (realtimeConfig.min_frames || 8) * (realtimeConfig.capture_interval_ms || 250),
      );
    }}

    async function captureOverlayFrameBlob() {{
      if (!cameraVideo || !cameraOverlayCaptureCanvas || !cameraStream || cameraVideo.readyState < 2) {{
        return null;
      }}
      const sourceWidth = cameraVideo.videoWidth || 640;
      const sourceHeight = cameraVideo.videoHeight || 480;
      const targetWidth = realtimeConfig.overlay_frame_width || 256;
      const targetHeight = Math.max(1, Math.round((sourceHeight / sourceWidth) * targetWidth));
      cameraOverlayCaptureCanvas.width = targetWidth;
      cameraOverlayCaptureCanvas.height = targetHeight;
      const ctx = cameraOverlayCaptureCanvas.getContext('2d');
      ctx.drawImage(cameraVideo, 0, 0, targetWidth, targetHeight);
      return await new Promise((resolve) => {{
        cameraOverlayCaptureCanvas.toBlob(
          (blob) => resolve(blob || null),
          'image/jpeg',
          realtimeConfig.overlay_jpeg_quality || 0.55,
        );
      }});
    }}

    function getRealtimeOverlayWsUrl() {{
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      return `${{protocol}}//${{window.location.host}}/ws/realtime/overlay`;
    }}

    function rejectRealtimeOverlaySocketPending(error) {{
      if (!realtimeOverlaySocketPending) {{
        return;
      }}
      const pending = realtimeOverlaySocketPending;
      realtimeOverlaySocketPending = null;
      if (pending.timeoutId) {{
        window.clearTimeout(pending.timeoutId);
      }}
      pending.reject(error instanceof Error ? error : new Error(String(error || '实时 pose WebSocket 已断开')));
    }}

    async function decodeRealtimeOverlayPacket(packetData) {{
      const buffer = packetData instanceof ArrayBuffer ? packetData : await packetData.arrayBuffer();
      if (!buffer || buffer.byteLength < 4) {{
        throw new Error('实时 pose 返回包过短。');
      }}
      const metaLength = new DataView(buffer).getUint32(0);
      if (metaLength < 0 || (4 + metaLength) > buffer.byteLength) {{
        throw new Error('实时 pose 返回包格式非法。');
      }}
      const metaBytes = new Uint8Array(buffer, 4, metaLength);
      const metaText = new TextDecoder('utf-8').decode(metaBytes);
      const meta = metaText ? JSON.parse(metaText) : {{}};
      const payloadOffset = 4 + metaLength;
      const overlayBytes = payloadOffset < buffer.byteLength ? buffer.slice(payloadOffset) : null;
      return {{
        meta,
        overlayBlob: overlayBytes && overlayBytes.byteLength > 0 ? new Blob([overlayBytes], {{ type: 'image/png' }}) : null,
      }};
    }}

    function closeRealtimeOverlaySocket(reason = '') {{
      const socket = realtimeOverlaySocket;
      realtimeOverlaySocket = null;
      realtimeOverlaySocketPromise = null;
      rejectRealtimeOverlaySocketPending(new Error(reason || '实时 pose WebSocket 已关闭'));
      if (!socket) {{
        return;
      }}
      socket.onopen = null;
      socket.onmessage = null;
      socket.onerror = null;
      socket.onclose = null;
      try {{
        if (socket.readyState === WebSocket.CONNECTING || socket.readyState === WebSocket.OPEN) {{
          socket.close(1000, 'client-close');
        }}
      }} catch (error) {{
      }}
    }}

    function buildRealtimeOverlayResult(meta, overlayBlob) {{
      return {{
        meta: meta || {{}},
        overlayBlob: overlayBlob || null,
      }};
    }}

    async function sendRealtimeOverlayFrameHttp(frameBlob) {{
      const resp = await fetch('/api/realtime/overlay', {{
        method: 'POST',
        headers: {{ 'Content-Type': 'application/octet-stream' }},
        body: frameBlob,
      }});
      const poseMessage = decodeURIComponent(resp.headers.get('X-Pose-Message') || '');
      const hasPose = resp.headers.get('X-Has-Pose') === '1';
      const meta = {{
        has_pose: hasPose,
        message: poseMessage,
        reused_previous_pose: resp.headers.get('X-Pose-Reused') === '1',
        tracking_mode: resp.headers.get('X-Pose-Tracking-Mode') || '',
        error: false,
      }};
      if (!resp.ok && resp.status !== 204) {{
        const errorText = (await resp.text()).trim();
        meta.error = true;
        meta.message = errorText || poseMessage || '实时 pose 更新失败';
        return buildRealtimeOverlayResult(meta, null);
      }}
      const overlayBlob = hasPose && resp.status !== 204 ? await resp.blob() : null;
      return buildRealtimeOverlayResult(meta, overlayBlob);
    }}

    async function sendRealtimeOverlayFrameWithFallback(frameBlob) {{
      if (realtimeOverlayTransport === 'http') {{
        return await sendRealtimeOverlayFrameHttp(frameBlob);
      }}
      try {{
        return await sendRealtimeOverlayFrame(frameBlob);
      }} catch (error) {{
        realtimeOverlayTransport = 'http';
        realtimeOverlayWsDisabledReason = error instanceof Error ? error.message : String(error || '实时 pose WebSocket 不可用');
        closeRealtimeOverlaySocket(realtimeOverlayWsDisabledReason);
        return await sendRealtimeOverlayFrameHttp(frameBlob);
      }}
    }}

    function ensureRealtimeOverlaySocket() {{
      if (!cameraStream) {{
        return Promise.reject(new Error('摄像头未开启。'));
      }}
      if (typeof WebSocket === 'undefined') {{
        return Promise.reject(new Error('当前浏览器不支持 WebSocket。'));
      }}
      if (realtimeOverlaySocket && realtimeOverlaySocket.readyState === WebSocket.OPEN) {{
        return Promise.resolve(realtimeOverlaySocket);
      }}
      if (realtimeOverlaySocketPromise) {{
        return realtimeOverlaySocketPromise;
      }}

      realtimeOverlaySocketPromise = new Promise((resolve, reject) => {{
        const socket = new WebSocket(getRealtimeOverlayWsUrl());
        let settled = false;
        socket.binaryType = 'arraybuffer';
        realtimeOverlaySocket = socket;

        const failConnection = (message) => {{
          const error = message instanceof Error ? message : new Error(String(message || '实时 pose WebSocket 连接失败'));
          if (realtimeOverlaySocket === socket) {{
            realtimeOverlaySocket = null;
          }}
          realtimeOverlaySocketPromise = null;
          rejectRealtimeOverlaySocketPending(error);
          if (!settled) {{
            settled = true;
            reject(error);
          }}
        }};

        socket.onopen = () => {{
          realtimeOverlaySocket = socket;
          if (!settled) {{
            settled = true;
            resolve(socket);
          }}
        }};

        socket.onmessage = async (event) => {{
          if (!realtimeOverlaySocketPending) {{
            return;
          }}
          const pending = realtimeOverlaySocketPending;
          realtimeOverlaySocketPending = null;
          if (pending.timeoutId) {{
            window.clearTimeout(pending.timeoutId);
          }}
          try {{
            pending.resolve(await decodeRealtimeOverlayPacket(event.data));
          }} catch (error) {{
            pending.reject(error instanceof Error ? error : new Error(String(error || '实时 pose 返回包解析失败')));
          }}
        }};

        socket.onerror = () => {{
          failConnection('实时 pose WebSocket 连接异常。');
        }};

        socket.onclose = (event) => {{
          const closeReason = event && event.reason ? event.reason : `连接已关闭（${{(event && event.code) || 1000}}）`;
          failConnection(`实时 pose WebSocket ${{closeReason}}`);
        }};
      }});

      return realtimeOverlaySocketPromise;
    }}

    async function sendRealtimeOverlayFrame(frameBlob) {{
      const socket = await ensureRealtimeOverlaySocket();
      if (realtimeOverlaySocketPending) {{
        throw new Error('上一帧 pose 结果仍在返回中。');
      }}
      return await new Promise((resolve, reject) => {{
        const timeoutId = window.setTimeout(() => {{
          if (!realtimeOverlaySocketPending || realtimeOverlaySocketPending.timeoutId !== timeoutId) {{
            return;
          }}
          realtimeOverlaySocketPending = null;
          try {{
            socket.close(1011, 'overlay-timeout');
          }} catch (error) {{
          }}
          reject(new Error('实时 pose WebSocket 响应超时。'));
        }}, Math.max(1500, (realtimeConfig.overlay_tracking_stale_ms || 700) + 1200));

        realtimeOverlaySocketPending = {{ resolve, reject, timeoutId }};
        try {{
          socket.send(frameBlob);
        }} catch (error) {{
          window.clearTimeout(timeoutId);
          realtimeOverlaySocketPending = null;
          reject(error instanceof Error ? error : new Error(String(error || '实时 pose 发送失败')));
        }}
      }});
    }}

    function getRealtimeRecordingMimeType() {{
      if (typeof MediaRecorder === 'undefined' || !MediaRecorder.isTypeSupported) {{
        return '';
      }}
      const candidates = ['video/webm;codecs=vp8', 'video/webm', 'video/webm;codecs=vp9'];
      return candidates.find((type) => MediaRecorder.isTypeSupported(type)) || '';
    }}

    async function flushRealtimeRecording() {{
      if (!compositeRecorder || compositeRecorder.state !== 'recording' || typeof compositeRecorder.requestData !== 'function') {{
        return;
      }}
      const beforeCount = compositeChunks.length;
      try {{
        compositeRecorder.requestData();
      }} catch (error) {{
        return;
      }}
      const deadline = Date.now() + 400;
      while (Date.now() < deadline) {{
        if (compositeChunks.length > beforeCount) {{
          return;
        }}
        await new Promise((resolve) => window.setTimeout(resolve, 40));
      }}
    }}

    function pruneRealtimeRecordingChunks(referenceTime = Date.now()) {{
      const maxAgeMs = getRealtimeWindowMs() + 8000;
      compositeChunks = compositeChunks.filter((chunk) => chunk.at >= referenceTime - maxAgeMs);
    }}

    async function buildRealtimeRecordingUrl(referenceTime = Date.now()) {{
      await flushRealtimeRecording();
      pruneRealtimeRecordingChunks(referenceTime);
      const startAt = referenceTime - getRealtimeWindowMs() - 1500;
      const selectedChunks = compositeChunks.filter((chunk) => chunk.at >= startAt && chunk.at <= referenceTime + 1000);
      if (!selectedChunks.length) {{
        return '';
      }}
      const mimeType = compositeRecorderMimeType || (compositeRecorder && compositeRecorder.mimeType) || 'video/webm';
      return URL.createObjectURL(new Blob(selectedChunks.map((chunk) => chunk.data), {{ type: mimeType }}));
    }}

    function stopRealtimeCanvasLoop() {{
      if (cameraRenderFrameId) {{
        window.cancelAnimationFrame(cameraRenderFrameId);
        cameraRenderFrameId = 0;
      }}
      if (cameraPreviewCanvas) {{
        const ctx = cameraPreviewCanvas.getContext('2d');
        ctx.clearRect(0, 0, cameraPreviewCanvas.width || 1, cameraPreviewCanvas.height || 1);
      }}
    }}

    function drawRealtimePreview() {{
      if (!cameraPreviewCanvas || !cameraVideo) {{
        return;
      }}
      const width = cameraVideo.videoWidth || cameraCaptureCanvas.width || realtimeConfig.frame_width || 320;
      const height = cameraVideo.videoHeight || cameraCaptureCanvas.height || Math.max(1, Math.round((3 * width) / 4));
      if (cameraPreviewCanvas.width !== width) {{
        cameraPreviewCanvas.width = width;
      }}
      if (cameraPreviewCanvas.height !== height) {{
        cameraPreviewCanvas.height = height;
      }}
      const previewCtx = cameraPreviewCanvas.getContext('2d');
      previewCtx.clearRect(0, 0, width, height);
      if (cameraVideo.readyState >= 2) {{
        previewCtx.drawImage(cameraVideo, 0, 0, width, height);
      }}
      if (cameraOverlay && !cameraOverlay.hidden && cameraOverlay.complete && cameraOverlay.naturalWidth > 0) {{
        previewCtx.drawImage(cameraOverlay, 0, 0, width, height);
      }}
      cameraRenderFrameId = window.requestAnimationFrame(drawRealtimePreview);
    }}

    function startRealtimeCanvasLoop() {{
      stopRealtimeCanvasLoop();
      cameraRenderFrameId = window.requestAnimationFrame(drawRealtimePreview);
    }}

    function stopRealtimeRecording() {{
      if (compositeRecorder && compositeRecorder.state !== 'inactive') {{
        compositeRecorder.stop();
      }}
      compositeRecorder = null;
      if (compositeRecorderStream) {{
        for (const track of compositeRecorderStream.getTracks()) {{
          track.stop();
        }}
        compositeRecorderStream = null;
      }}
    }}

    function startRealtimeRecording() {{
      stopRealtimeRecording();
      if (!cameraPreviewCanvas || typeof cameraPreviewCanvas.captureStream !== 'function' || typeof MediaRecorder === 'undefined') {{
        return;
      }}
      compositeRecorderMimeType = getRealtimeRecordingMimeType();
      const stream = cameraPreviewCanvas.captureStream(Math.max(10, Math.round(1000 / (realtimeConfig.capture_interval_ms || 250))));
      try {{
        compositeRecorder = compositeRecorderMimeType
          ? new MediaRecorder(stream, {{ mimeType: compositeRecorderMimeType }})
          : new MediaRecorder(stream);
      }} catch (error) {{
        for (const track of stream.getTracks()) {{
          track.stop();
        }}
        compositeRecorder = null;
        compositeRecorderStream = null;
        return;
      }}
      compositeRecorderStream = stream;
      compositeChunks = [];
      compositeRecorder.ondataavailable = (event) => {{
        if (!event.data || event.data.size <= 0) {{
          return;
        }}
        compositeChunks.push({{ at: Date.now(), data: event.data }});
        pruneRealtimeRecordingChunks();
      }};
      compositeRecorder.start(250);
    }}

    function stopRealtimeOverlayPump() {{
      if (overlayPumpTimer) {{
        window.clearTimeout(overlayPumpTimer);
        overlayPumpTimer = null;
      }}
    }}

    function scheduleRealtimeOverlayPump(delayMs = 0) {{
      stopRealtimeOverlayPump();
      if (!cameraStream) {{
        return;
      }}
      overlayPumpTimer = window.setTimeout(() => {{
        overlayPumpTimer = null;
        requestRealtimeOverlay();
      }}, Math.max(0, delayMs));
    }}

    async function requestRealtimeOverlay(frameBlob = null) {{
      if (realtimeOverlayBusy || !cameraStream) {{
        return;
      }}
      const overlayFrame = frameBlob || await captureOverlayFrameBlob();
      if (!overlayFrame) {{
        scheduleRealtimeOverlayPump(16);
        return;
      }}
      realtimeOverlayBusy = true;
      lastRealtimeOverlayAt = Date.now();
      const requestSeq = ++realtimeOverlayRequestSeq;
      try {{
        const {{ meta, overlayBlob }} = await sendRealtimeOverlayFrameWithFallback(overlayFrame);
        if (requestSeq < realtimeOverlayAppliedSeq) {{
          return;
        }}
        realtimeOverlayAppliedSeq = requestSeq;
        const poseMessage = String((meta && meta.message) || '');
        if (meta && meta.error) {{
          throw new Error(poseMessage || '实时 pose 更新失败');
        }}
        const hasPose = Boolean(meta && meta.has_pose) && overlayBlob;
        if (!hasPose) {{
          setRealtimeOverlay('');
          setRealtimeIssue(poseMessage || '当前画面未检测到有效 pose，请调整站位、光线和入镜范围。');
        }} else {{
          const overlayUrl = URL.createObjectURL(overlayBlob);
          setRealtimeOverlay(overlayUrl);
          setRealtimeIssue('');
        }}
      }} catch (error) {{
        if (!cameraStream) {{
          return;
        }}
        setRealtimeOverlay('');
        setRealtimeIssue(`实时 pose 更新失败：${{error.message}}`);
        updateRealtimeCaptureStatus();
      }} finally {{
        realtimeOverlayBusy = false;
        if (cameraStream) {{
          scheduleRealtimeOverlayPump(0);
        }}
      }}
    }}

    function updateRealtimeCaptureStatus() {{
      if (!cameraStream) {{
        return;
      }}
      const minFrames = realtimeConfig.min_frames || 8;
      const windowFrames = realtimeFrames.length;
      const maxBufferedFrames = realtimeConfig.max_buffered_frames || 32;
      const issueSuffix = realtimeIssueMessage ? ` 提示：${{realtimeIssueMessage}}` : '';
      if (realtimeBusy) {{
        const elapsedSeconds = realtimeInferenceStartedAt ? Math.max(1, Math.round((Date.now() - realtimeInferenceStartedAt) / 1000)) : 0;
        const deviceText = realtimeConfig.runtime_device === 'cuda' ? 'GPU' : 'CPU';
        setRealtimeHint(`滑窗帧：${{windowFrames}} / ${{maxBufferedFrames}}，累计采样：${{totalCapturedFrames}} 帧，阈值：${{minFrames}} 帧，正在执行实时推理...（${{deviceText}}，已等待 ${{elapsedSeconds}} 秒）`);
        return;
      }}
      if (windowFrames < minFrames) {{
        setRealtimeHint(`正在积累实时帧：滑窗 ${{windowFrames}} / ${{minFrames}}，累计采样 ${{totalCapturedFrames}} 帧。${{issueSuffix}}`);
        return;
      }}
      const overlayLagMs = lastRealtimeOverlayAt ? Math.max(0, Date.now() - lastRealtimeOverlayAt) : 0;
      setRealtimeHint(`滑窗帧：${{windowFrames}} / ${{maxBufferedFrames}}，累计采样：${{totalCapturedFrames}} 帧，已达到阈值 ${{minFrames}} 帧，后台等待下一次字幕推理；pose 叠加持续刷新中（最近更新约 ${{overlayLagMs}} ms 前）。${{issueSuffix}}`);
    }}

    function stopRealtimeStatusTimer() {{
      if (realtimeStatusTimer) {{
        window.clearInterval(realtimeStatusTimer);
        realtimeStatusTimer = null;
      }}
    }}

    function startRealtimeStatusTimer() {{
      stopRealtimeStatusTimer();
      realtimeStatusTimer = window.setInterval(() => {{
        if (realtimeBusy) {{
          updateRealtimeCaptureStatus();
        }}
      }}, 1000);
    }}

    function getSelectedSample() {{
      if (!sampleSelect) {{
        return null;
      }}
      return defaultSamples.find((sample) => sample.id === sampleSelect.value) || defaultSamples[0] || null;
    }}

    function updateSamplePreview() {{
      const sample = getSelectedSample();
      if (!sample || !samplePreview || !samplePreviewSource || !sampleMeta) {{
        return;
      }}
      samplePreviewSource.src = sample.preview_url;
      sampleMeta.textContent = `当前样本：${{sample.filename}}`;
      samplePreview.load();
    }}

    function maybeRunRealtimeInference() {{
      if (!cameraStream || realtimeBusy) {{
        return;
      }}
      const inferenceIntervalMs = realtimeConfig.inference_interval_ms || 2000;
      const minFrames = realtimeConfig.min_frames || 8;
      if (realtimeFrames.length < minFrames) {{
        return;
      }}
      if (Date.now() - lastRealtimeInferenceAt < inferenceIntervalMs) {{
        return;
      }}
      window.setTimeout(() => {{
        if (cameraStream && !realtimeBusy) {{
          requestRealtimeInference();
        }}
      }}, 0);
    }}

    function captureRealtimeFrame() {{
      if (!cameraVideo || !cameraCaptureCanvas || cameraVideo.readyState < 2) {{
        return;
      }}
      const sourceWidth = cameraVideo.videoWidth || 640;
      const sourceHeight = cameraVideo.videoHeight || 480;
      const targetWidth = realtimeConfig.frame_width || 320;
      const targetHeight = Math.max(1, Math.round((sourceHeight / sourceWidth) * targetWidth));
      cameraCaptureCanvas.width = targetWidth;
      cameraCaptureCanvas.height = targetHeight;
      const ctx = cameraCaptureCanvas.getContext('2d');
      ctx.drawImage(cameraVideo, 0, 0, targetWidth, targetHeight);
      latestCapturedFrame = cameraCaptureCanvas.toDataURL('image/jpeg', realtimeConfig.jpeg_quality || 0.7);
      realtimeFrames.push(latestCapturedFrame);
      totalCapturedFrames += 1;
      const maxBufferedFrames = realtimeConfig.max_buffered_frames || 32;
      if (realtimeFrames.length > maxBufferedFrames) {{
        realtimeFrames = realtimeFrames.slice(-maxBufferedFrames);
      }}
      updateRealtimeCaptureStatus();
    }}

    async function requestRealtimeInference() {{
      if (realtimeBusy) {{
        return;
      }}
      if (!cameraStream) {{
        return;
      }}
      const minFrames = realtimeConfig.min_frames || 8;
      if (realtimeFrames.length < minFrames) {{
        updateRealtimeCaptureStatus();
        return;
      }}
      realtimeBusy = true;
      realtimeInferenceStartedAt = Date.now();
      const inferenceRequestedAt = Date.now();
      lastRealtimeInferenceAt = inferenceRequestedAt;
      const framesSnapshot = realtimeFrames.slice();
      try {{
        startRealtimeStatusTimer();
        setRealtimeIssue('');
        const deviceText = realtimeConfig.runtime_device === 'cuda' ? 'GPU' : 'CPU';
        setRealtimeHint(`滑窗帧：${{framesSnapshot.length}} / ${{realtimeConfig.max_buffered_frames || 32}}，累计采样：${{totalCapturedFrames}} 帧，阈值：${{minFrames}} 帧，后台正在发送字幕推理请求...（${{deviceText}}）`);
        const resp = await fetch('/api/realtime/infer', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ frames: framesSnapshot, window_seconds: realtimeConfig.window_seconds }}),
        }});
        const data = await readApiResponse(resp);
        if (!resp.ok) {{
          throw new Error(data.error || '实时推理失败');
        }}
        setRealtimeSubtitle(data.result || '当前未识别到稳定字幕。');
        setRealtimeIssue(data.has_pose === false ? (data.message || '当前滑窗未检测到稳定 pose，请调整站位、光线和入镜范围。') : '');
        setRealtimeHint(data.message || '后台字幕推理已完成，pose 跟踪继续刷新。');
        const realtimeClipUrl = await buildRealtimeRecordingUrl(inferenceRequestedAt);
        ensureRealtimeClipCard(data.result || '', realtimeClipUrl);
      }} catch (error) {{
        setRealtimeIssue(`最近一次实时推理失败：${{error.message}}`);
        setRealtimeSubtitle('实时推理失败，请根据状态提示调整后重试。');
        setRealtimeHint(`实时推理异常：${{error.message}}`);
      }} finally {{
        realtimeBusy = false;
        realtimeInferenceStartedAt = 0;
        stopRealtimeStatusTimer();
        updateRealtimeCaptureStatus();
      }}
    }}

    async function startCamera() {{
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {{
        setRealtimeHint('当前浏览器不支持摄像头访问。');
        return;
      }}
      if (cameraStream) {{
        setRealtimeHint('摄像头已经开启。');
        return;
      }}
      try {{
        cameraStream = await navigator.mediaDevices.getUserMedia({{ video: true, audio: false }});
        cameraVideo.srcObject = cameraStream;
        if (cameraVideo.play) {{
          await cameraVideo.play().catch(() => null);
        }}
        closeRealtimeOverlaySocket('摄像头重新初始化。');
        realtimeFrames = [];
        totalCapturedFrames = 0;
        lastRealtimeInferenceAt = 0;
        latestCapturedFrame = '';
        realtimeOverlayBusy = false;
        realtimeOverlayTransport = 'websocket';
        realtimeOverlayWsDisabledReason = '';
        lastRealtimeOverlayAt = 0;
        realtimeOverlayRequestSeq = 0;
        realtimeOverlayAppliedSeq = 0;
        realtimeInferenceStartedAt = 0;
        setRealtimeIssue('');
        setRealtimeOverlay('');
        setRealtimeSubtitle('摄像头已打开，后台等待首批实时字幕。');
        const deviceText = realtimeConfig.runtime_device === 'cuda' ? 'GPU' : 'CPU';
        setRealtimeHint(`已开启摄像头，单窗口 pose 会按当前设备可承受的最高速率刷新；后台约每 ${{Math.round((realtimeConfig.inference_interval_ms || 2000) / 1000)}} 秒做一次字幕推理，当前设备：${{deviceText}}。`);
        startRealtimeCanvasLoop();
        startRealtimeRecording();
        captureTimer = window.setInterval(captureRealtimeFrame, realtimeConfig.capture_interval_ms || 250);
        inferTimer = window.setInterval(maybeRunRealtimeInference, Math.max(250, Math.min(1000, realtimeConfig.capture_interval_ms || 250)));
        scheduleRealtimeOverlayPump(0);
        maybeRunRealtimeInference();
      }} catch (error) {{
        setRealtimeHint(`打开摄像头失败：${{error.message}}`);
      }}
    }}

    function stopCamera() {{
      if (captureTimer) {{
        window.clearInterval(captureTimer);
        captureTimer = null;
      }}
      if (inferTimer) {{
        window.clearInterval(inferTimer);
        inferTimer = null;
      }}
      stopRealtimeOverlayPump();
      closeRealtimeOverlaySocket('摄像头已停止。');
      realtimeFrames = [];
      totalCapturedFrames = 0;
      realtimeBusy = false;
      lastRealtimeInferenceAt = 0;
      latestCapturedFrame = '';
      realtimeOverlayBusy = false;
      realtimeOverlayTransport = 'websocket';
      realtimeOverlayWsDisabledReason = '';
      lastRealtimeOverlayAt = 0;
      realtimeOverlayRequestSeq = 0;
      realtimeOverlayAppliedSeq = 0;
      realtimeInferenceStartedAt = 0;
      setRealtimeIssue('');
      stopRealtimeStatusTimer();
      if (cameraStream) {{
        for (const track of cameraStream.getTracks()) {{
          track.stop();
        }}
        cameraStream = null;
      }}
      if (cameraVideo) {{
        cameraVideo.srcObject = null;
      }}
      stopRealtimeRecording();
      stopRealtimeCanvasLoop();
      setRealtimeOverlay('');
      setRealtimeHint('摄像头已停止。');
      setRealtimeSubtitle('等待摄像头启动。');
    }}

    async function pollJob(jobId, uploadId, filename) {{
      while (true) {{
        const query = jobId
          ? `job_id=${{encodeURIComponent(jobId)}}`
          : `upload_id=${{encodeURIComponent(uploadId)}}`;
        const resp = await fetch(`/api/job?${{query}}`, {{
          method: 'GET',
          cache: 'no-store',
        }});
        const data = await readApiResponse(resp);
        if (!resp.ok) {{
          throw new Error(data.error || '查询任务状态失败');
        }}

        if (data.status === 'queued' || data.status === 'running') {{
          setStatus(data.message || '正在后台处理，请稍候...');
          await new Promise((resolve) => setTimeout(resolve, 2000));
          continue;
        }}

        if (data.status === 'failed') {{
          throw new Error(data.error || data.message || '推理失败');
        }}

        if (data.status === 'completed') {{
          setStatus(`推理完成：${{filename}}` + (data.inference_seconds ? `（模型推理约 ${{data.inference_seconds}} 秒）` : ''));
          return data;
        }}

        throw new Error('未知任务状态。');
      }}
    }}

    if (sampleSelect) {{
      sampleSelect.addEventListener('change', updateSamplePreview);
      updateSamplePreview();
    }}

    if (sampleRunButton) {{
      sampleRunButton.addEventListener('click', async () => {{
        const sample = getSelectedSample();
        if (!sample) {{
          setStatus('当前没有可用的默认样本。');
          return;
        }}

        try {{
          setStatus(`正在使用默认样本推理：${{sample.filename}}`);
          const resp = await fetch('/api/sample/start', {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ sample_id: sample.id }}),
          }});
          const data = await readApiResponse(resp);
          if (!resp.ok) {{
            throw new Error(data.error || '默认样本推理启动失败');
          }}
          const finalData = await pollJob(data.job_id || '', '', data.filename || sample.filename);
          if (finalData.pose_video_url) {{
            ensureVideoCard(finalData.filename || sample.filename, finalData.pose_video_url);
          }}
          ensureResultCard('推理结果', finalData.filename || sample.filename, finalData.result, 'success');
        }} catch (error) {{
          setStatus(`处理失败：${{error.message}}`);
          ensureResultCard('处理失败', sample.filename, error.message, 'error');
        }}
      }});
    }}

    if (startCameraButton) {{
      startCameraButton.addEventListener('click', startCamera);
    }}

    if (stopCameraButton) {{
      stopCameraButton.addEventListener('click', stopCamera);
    }}

    if (cameraOverlay) {{
      cameraOverlay.addEventListener('error', () => setRealtimeOverlay(''));
    }}

    window.addEventListener('beforeunload', stopCamera);

    uploadForm.addEventListener('submit', async (event) => {{
      event.preventDefault();
      const file = videoInput.files[0];
      if (!file) {{
        setStatus('请先选择一个视频文件。');
        return;
      }}

      let uploadId = '';
      try {{
        setStatus(`初始化上传：${{file.name}} (${{(file.size / 1024 / 1024).toFixed(1)}} MB)`);
        const initResp = await fetch('/api/upload/init', {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{
            filename: file.name,
            size: file.size,
            content_type: file.type || 'application/octet-stream',
          }}),
        }});
        const initData = await readApiResponse(initResp);
        if (!initResp.ok) {{
          throw new Error(initData.error || '初始化上传失败');
        }}
        uploadId = initData.upload_id || initData.uploadId || '';
        if (!uploadId) {{
          throw new Error('初始化上传成功，但服务端没有返回 upload_id。');
        }}

        const totalChunks = Math.max(1, Math.ceil(file.size / chunkSize));
        for (let index = 0; index < totalChunks; index += 1) {{
          const start = index * chunkSize;
          const end = Math.min(file.size, start + chunkSize);
          const chunk = file.slice(start, end);
          setStatus(`正在上传分片 ${{index + 1}}/${{totalChunks}}...`);
          const chunkResp = await fetch(`/api/upload/chunk?upload_id=${{encodeURIComponent(uploadId)}}&index=${{index}}`, {{
            method: 'POST',
            headers: {{
              'Content-Type': 'application/octet-stream',
              'X-Upload-Id': uploadId,
              'X-Chunk-Index': String(index),
            }},
            body: chunk,
          }});
          const chunkData = await readApiResponse(chunkResp);
          if (!chunkResp.ok) {{
            throw new Error(chunkData.error || '上传分片失败');
          }}
          const percent = chunkData.total_size ? (chunkData.received / chunkData.total_size) * 100 : 100;
          setStatus(`上传完成 ${{percent.toFixed(1)}}%，等待继续处理...`);
        }}

        setStatus('上传完成，正在执行 Uni-Sign 推理，请稍候...');
        const completeResp = await fetch(`/api/upload/complete?upload_id=${{encodeURIComponent(uploadId)}}`, {{
          method: 'POST',
          headers: {{
            'Content-Type': 'application/json',
            'X-Upload-Id': uploadId,
          }},
          body: JSON.stringify({{ upload_id: uploadId }}),
        }});
        const completeData = await readApiResponse(completeResp);
        if (!completeResp.ok) {{
          throw new Error(completeData.error || '推理失败');
        }}

        const finalData = (completeData.job_id || completeData.upload_id || uploadId)
          ? await pollJob(completeData.job_id || '', completeData.upload_id || uploadId, file.name)
          : completeData;

        if (finalData.pose_video_url) {{
          ensureVideoCard(finalData.filename || file.name, finalData.pose_video_url);
        }}
        ensureResultCard('推理结果', finalData.filename || file.name, finalData.result, 'success');
      }} catch (error) {{
        setStatus(`处理失败：${{error.message}}`);
        ensureResultCard('处理失败', file.name, error.message, 'error');
      }}
    }});
  </script>
</body>
</html>
"""


class UniSignWebService:
    def __init__(self, args):
        self.args = args
        if not self.args.finetune:
            self.args.finetune = DEFAULT_CHECKPOINT
        utils.set_seed(self.args.seed)
        self.model = load_model(self.args)
        self.lock = threading.Lock()
        self.pose_lock = threading.Lock()
        self.overlay_pose_lock = threading.Lock()
        self.pose_infer_lock = threading.Lock()
        self.overlay_pose_infer_lock = threading.Lock()
        self.overlay_tracking_lock = threading.Lock()
        self.pose_estimator = None
        self.overlay_pose_estimator = None
        self.overlay_tracking_state = {
            "keypoints": None,
            "scores": None,
            "updated_at": 0.0,
            "bbox": None,
            "requests_since_full_frame": 0,
            "last_mode": "full",
        }
        self.uploads = UploadManager(
            max_upload_bytes=int(self.args.max_upload_size_mb * 1024 * 1024),
            max_chunk_bytes=int(self.args.chunk_size_mb * 1024 * 1024),
        )
        self.samples = SampleVideoManager(DEFAULT_SAMPLE_VIDEO_DIR)
        self.realtime_profile = self._build_realtime_profile()
        self.results = ResultManager()
        self.jobs = InferenceJobManager(self.run_inference_with_preview, self.uploads.cleanup, self.run_sample_inference)

    def _build_realtime_profile(self) -> dict:
        sample_summary = self.samples.summary()
        summary = dict(DEFAULT_CE_CSL_TRAINING_SUMMARY)
        summary["sample_avg_duration_seconds"] = float(sample_summary.get("avg_duration_seconds") or 0.0)
        runtime_device = get_runtime_device(self.args)
        window_seconds = max(4.0, min(12.0, float(summary.get("avg_duration_seconds") or 8.0)))
        target_capture_frames = 48 if runtime_device == "cuda" else 36
        min_capture_interval_ms = 100 if runtime_device == "cuda" else 120
        max_capture_interval_ms = 220 if runtime_device == "cuda" else 260
        capture_interval_ms = int(
            max(
                min_capture_interval_ms,
                min(max_capture_interval_ms, round((window_seconds * 1000) / target_capture_frames)),
            )
        )
        inference_interval_ms = (
            max(1400, int(window_seconds * 220))
            if runtime_device == "cuda"
            else max(6000, int(window_seconds * 1000))
        )
        max_buffered_frames = max(
            24 if runtime_device == "cuda" else 16,
            min(48 if runtime_device == "cuda" else 40, int(round(window_seconds * 1000 / capture_interval_ms))),
        )
        min_frames_ratio = 0.6 if runtime_device == "cuda" else 0.5
        min_frames_floor = 24 if runtime_device == "cuda" else 16
        min_frames = min(max(min_frames_floor, int(round(max_buffered_frames * min_frames_ratio))), max_buffered_frames)
        return {
            "runtime_device": runtime_device,
            "window_seconds": round(window_seconds, 2),
            "capture_interval_ms": capture_interval_ms,
            "overlay_interval_ms": max(45, min(75, int(round(capture_interval_ms * 0.45)))) if runtime_device == "cuda" else max(80, min(110, int(round(capture_interval_ms * 0.55)))),
            "inference_interval_ms": inference_interval_ms,
            "max_buffered_frames": max_buffered_frames,
            "min_frames": min_frames,
            "frame_width": 320,
            "jpeg_quality": 0.7,
            "overlay_frame_width": 256 if runtime_device == "cuda" else 160,
            "overlay_jpeg_quality": 0.55 if runtime_device == "cuda" else 0.4,
            "max_inference_frames": min(40 if runtime_device == "cuda" else 32, max_buffered_frames),
            "overlay_radius": 1,
            "overlay_line_width": 1,
            "overlay_smoothing_alpha": 0.65 if runtime_device == "cuda" else 0.3,
            "overlay_hold_ms": 220 if runtime_device == "cuda" else 120,
            "overlay_tracking_stale_ms": 900 if runtime_device == "cuda" else 700,
            "overlay_full_frame_interval": 4 if runtime_device == "cuda" else 3,
            "overlay_track_margin_ratio": 0.25 if runtime_device == "cuda" else 0.2,
            "overlay_min_crop_size": 160 if runtime_device == "cuda" else 112,
        }

    def predict(self, video_path: str, pose_data=None) -> str:
        with self.lock:
            return predict_video(video_path, self.model, self.args, pose_data=pose_data)

    def decode_realtime_frames(self, frames: list[str]) -> list[np.ndarray]:
        decoded_frames = []
        for frame_data in frames:
            if not frame_data:
                continue
            payload = frame_data.split(",", 1)[-1]
            frame_bytes = base64.b64decode(payload)
            buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
            if frame is not None:
                decoded_frames.append(frame)
        return decoded_frames

    def extract_pose_from_frames(self, frames: list[np.ndarray], max_frames: int = 0) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
        if not frames:
            raise ValueError("实时推理没有收到有效视频帧。")

        sampled_frames = frames
        if max_frames and len(sampled_frames) > max_frames:
            indices = np.linspace(0, len(sampled_frames) - 1, max_frames).round().astype(int).tolist()
            sampled_frames = [sampled_frames[index] for index in indices]

        pose_data = {"keypoints": [], "scores": []}
        latest_pose_frame = sampled_frames[-1]
        latest_keypoints = None
        latest_scores = None

        for frame in sampled_frames:
            keypoints, scores = self._estimate_pose(frame)
            height, width = frame.shape[:2]
            pose_data["keypoints"].append(keypoints / np.array([width, height])[None, None])
            pose_data["scores"].append(scores)
            latest_pose_frame = frame
            latest_keypoints = keypoints
            latest_scores = scores

        return pose_data, latest_pose_frame, latest_keypoints, latest_scores

    @staticmethod
    def decode_realtime_frame_bytes(frame_bytes: bytes) -> Optional[np.ndarray]:
        if not frame_bytes:
            return None
        buffer = np.frombuffer(frame_bytes, dtype=np.uint8)
        return cv2.imdecode(buffer, cv2.IMREAD_COLOR)

    @staticmethod
    def _select_valid_tracking_points(keypoints_array: np.ndarray, scores_array: np.ndarray, threshold: float = 0.3) -> tuple[Optional[np.ndarray], str]:
        if keypoints_array.ndim != 2 or scores_array.ndim != 1 or keypoints_array.shape[0] != scores_array.shape[0]:
            return None, "invalid"

        total_points = keypoints_array.shape[0]
        valid_mask = scores_array > threshold
        if int(np.count_nonzero(valid_mask)) < 4:
            return None, "missing"

        hand_indices = [index for index in WHOLEBODY_HAND_KEYPOINT_INDICES if index < total_points]
        upper_body_indices = [index for index in WHOLEBODY_UPPER_BODY_KEYPOINT_INDICES if index < total_points]

        if hand_indices:
            hand_mask = np.zeros(total_points, dtype=bool)
            hand_mask[hand_indices] = True
            valid_hand_mask = valid_mask & hand_mask
            if int(np.count_nonzero(valid_hand_mask)) >= 6:
                upper_mask = np.zeros(total_points, dtype=bool)
                upper_mask[upper_body_indices] = True
                tracked_points = keypoints_array[valid_mask & upper_mask]
                if tracked_points.size:
                    return tracked_points, "hands"

        if upper_body_indices:
            upper_mask = np.zeros(total_points, dtype=bool)
            upper_mask[upper_body_indices] = True
            valid_upper_mask = valid_mask & upper_mask
            if int(np.count_nonzero(valid_upper_mask)) >= 4:
                tracked_points = keypoints_array[valid_upper_mask]
                if tracked_points.size:
                    return tracked_points, "upper_body"

        tracked_points = keypoints_array[valid_mask]
        return tracked_points if tracked_points.size else None, "full_body"

    def _compute_pose_bbox(self, keypoints, scores, frame_shape: tuple[int, int, int]) -> Optional[tuple[int, int, int, int]]:
        keypoints_array = np.asarray(keypoints, dtype=np.float32)
        scores_array = np.asarray(scores, dtype=np.float32)
        if keypoints_array.ndim == 3:
            keypoints_array = keypoints_array[0]
        if scores_array.ndim == 2:
            scores_array = scores_array[0]
        tracked_points, tracking_focus = self._select_valid_tracking_points(keypoints_array, scores_array, threshold=0.3)
        if tracked_points is None:
            return None
        valid_points = tracked_points
        min_x = float(np.min(valid_points[:, 0]))
        max_x = float(np.max(valid_points[:, 0]))
        min_y = float(np.min(valid_points[:, 1]))
        max_y = float(np.max(valid_points[:, 1]))
        margin_ratio = float(self.realtime_profile.get("overlay_track_margin_ratio", 0.2) or 0.2)
        if tracking_focus == "hands":
            margin_ratio *= 1.25
        elif tracking_focus == "full_body":
            margin_ratio *= 0.9
        min_crop_size = int(self.realtime_profile.get("overlay_min_crop_size", 112) or 112)
        box_w = max(1.0, max_x - min_x)
        box_h = max(1.0, max_y - min_y)
        expand = max(box_w, box_h) * margin_ratio
        height, width = frame_shape[:2]
        x1 = max(0, int(np.floor(min_x - expand)))
        y1 = max(0, int(np.floor(min_y - expand)))
        x2 = min(width, int(np.ceil(max_x + expand)))
        y2 = min(height, int(np.ceil(max_y + expand)))
        if (x2 - x1) < min_crop_size:
            center_x = int(round((x1 + x2) / 2))
            half = int(np.ceil(min_crop_size / 2))
            x1 = max(0, center_x - half)
            x2 = min(width, center_x + half)
        if (y2 - y1) < min_crop_size:
            center_y = int(round((y1 + y2) / 2))
            half = int(np.ceil(min_crop_size / 2))
            y1 = max(0, center_y - half)
            y2 = min(height, center_y + half)
        if x2 - x1 < 8 or y2 - y1 < 8:
            return None
        return x1, y1, x2, y2

    def _estimate_overlay_pose_tracked(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
        now = time.time()
        with self.overlay_tracking_lock:
            state_snapshot = dict(self.overlay_tracking_state)

        bbox = state_snapshot.get("bbox")
        last_updated_at = float(state_snapshot.get("updated_at") or 0.0)
        requests_since_full_frame = int(state_snapshot.get("requests_since_full_frame") or 0)
        stale_seconds = max(0.2, float(self.realtime_profile.get("overlay_tracking_stale_ms", 700) or 700) / 1000.0)
        full_frame_interval = max(1, int(self.realtime_profile.get("overlay_full_frame_interval", 3) or 3))

        can_use_roi = (
            isinstance(bbox, (tuple, list))
            and len(bbox) == 4
            and (now - last_updated_at) <= stale_seconds
            and requests_since_full_frame < full_frame_interval
        )
        if can_use_roi:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                keypoints, scores = self._estimate_overlay_pose(crop)
                keypoints = np.asarray(keypoints, dtype=np.float32).copy()
                keypoints[..., 0] += x1
                keypoints[..., 1] += y1
                if self._has_valid_pose(scores):
                    with self.overlay_tracking_lock:
                        self.overlay_tracking_state["requests_since_full_frame"] = requests_since_full_frame + 1
                        self.overlay_tracking_state["last_mode"] = "roi"
                    return keypoints, np.asarray(scores, dtype=np.float32), "roi"

        keypoints, scores = self._estimate_overlay_pose(frame)
        with self.overlay_tracking_lock:
            self.overlay_tracking_state["requests_since_full_frame"] = 0
            self.overlay_tracking_state["last_mode"] = "full"
        return np.asarray(keypoints, dtype=np.float32), np.asarray(scores, dtype=np.float32), "full"

    def encode_overlay_image_bytes(self, frame: np.ndarray, keypoints, scores) -> bytes:
        height, width = frame.shape[:2]
        pose_frame = draw_skeleton(
            np.zeros((height, width, 3), dtype=np.uint8),
            keypoints,
            scores,
            openpose_skeleton=False,
            kpt_thr=0.3,
            radius=max(1, int(self.realtime_profile.get("overlay_radius", 1))),
            line_width=max(1, int(self.realtime_profile.get("overlay_line_width", 1))),
        )
        alpha = np.where(np.any(pose_frame > 0, axis=2), 255, 0).astype(np.uint8)
        pose_frame = cv2.cvtColor(pose_frame, cv2.COLOR_BGR2BGRA)
        pose_frame[:, :, 3] = alpha
        ok, buffer = cv2.imencode(".png", pose_frame)
        if not ok:
            raise ValueError("实时 pose 覆盖层编码失败。")
        return buffer.tobytes()

    def encode_overlay_image(self, frame: np.ndarray, keypoints, scores) -> str:
        return "data:image/png;base64," + base64.b64encode(self.encode_overlay_image_bytes(frame, keypoints, scores)).decode("utf-8")

    @staticmethod
    def _has_valid_pose(scores, threshold: float = 0.3) -> bool:
        if scores is None:
            return False
        scores_array = np.asarray(scores)
        return bool(scores_array.size and np.any(scores_array > threshold))

    def count_valid_pose_frames(self, pose_data: dict, threshold: float = 0.3) -> int:
        valid_frames = 0
        for scores in pose_data.get("scores") or []:
            if self._has_valid_pose(scores, threshold=threshold):
                valid_frames += 1
        return valid_frames

    def _stabilize_overlay_pose(self, keypoints, scores, frame_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray, bool]:
        keypoints_array = np.asarray(keypoints, dtype=np.float32)
        scores_array = np.asarray(scores, dtype=np.float32)
        now = time.time()
        smoothing_alpha = float(self.realtime_profile.get("overlay_smoothing_alpha", 0.0) or 0.0)
        hold_seconds = max(0.0, float(self.realtime_profile.get("overlay_hold_ms", 0.0) or 0.0) / 1000.0)
        has_pose = self._has_valid_pose(scores_array)

        with self.overlay_tracking_lock:
            previous_state = dict(self.overlay_tracking_state)
            last_keypoints = self.overlay_tracking_state.get("keypoints")
            last_scores = self.overlay_tracking_state.get("scores")
            last_updated_at = float(self.overlay_tracking_state.get("updated_at") or 0.0)
            recently_tracked = (now - last_updated_at) <= max(hold_seconds, 0.5)

            if has_pose:
                if (
                    smoothing_alpha > 0.0
                    and recently_tracked
                    and isinstance(last_keypoints, np.ndarray)
                    and isinstance(last_scores, np.ndarray)
                    and last_keypoints.shape == keypoints_array.shape
                    and last_scores.shape == scores_array.shape
                ):
                    keypoints_array = smoothing_alpha * keypoints_array + (1.0 - smoothing_alpha) * last_keypoints
                    scores_array = np.maximum(scores_array, (1.0 - smoothing_alpha) * last_scores + smoothing_alpha * scores_array)

                self.overlay_tracking_state = {
                    "keypoints": keypoints_array.copy(),
                    "scores": scores_array.copy(),
                    "updated_at": now,
                    "bbox": self._compute_pose_bbox(keypoints_array, scores_array, frame_shape),
                    "requests_since_full_frame": int(previous_state.get("requests_since_full_frame") or 0),
                    "last_mode": previous_state.get("last_mode", "full"),
                }
                return keypoints_array, scores_array, False

            if (
                hold_seconds > 0.0
                and recently_tracked
                and isinstance(last_keypoints, np.ndarray)
                and isinstance(last_scores, np.ndarray)
                and self._has_valid_pose(last_scores)
            ):
                return last_keypoints.copy(), last_scores.copy(), True

            self.overlay_tracking_state = {
                "keypoints": None,
                "scores": None,
                "updated_at": 0.0,
                "bbox": None,
                "requests_since_full_frame": int(previous_state.get("requests_since_full_frame") or 0),
                "last_mode": previous_state.get("last_mode", "full"),
            }
            return keypoints_array, scores_array, False

    def _run_realtime_overlay_frame(self, latest_frame: np.ndarray) -> dict:
        keypoints, scores, tracking_mode = self._estimate_overlay_pose_tracked(latest_frame)
        keypoints, scores, reused_previous_pose = self._stabilize_overlay_pose(keypoints, scores, latest_frame.shape)
        if not self._has_valid_pose(scores):
            return {
                "overlay_png_bytes": b"",
                "has_pose": False,
                "reused_previous_pose": False,
                "tracking_mode": tracking_mode,
                "message": "当前画面未检测到清晰 pose，请保持全身尤其双手在画面内，并确保光线充足。",
            }
        if reused_previous_pose:
            message = "沿用最近一帧 pose，保持跟踪连续性。"
        elif tracking_mode == "roi":
            message = "局部 ROI 跟踪已更新。"
        else:
            message = "全图 pose 覆盖层已更新。"
        return {
            "overlay_png_bytes": self.encode_overlay_image_bytes(latest_frame, keypoints, scores),
            "has_pose": True,
            "reused_previous_pose": reused_previous_pose,
            "tracking_mode": tracking_mode,
            "message": message,
        }

    def run_realtime_overlay(self, frame_data: str) -> dict:
        decoded_frames = self.decode_realtime_frames([frame_data])
        if not decoded_frames:
            raise ValueError("实时 pose 覆盖层没有收到有效视频帧。")
        latest_frame = decoded_frames[-1]
        output = self._run_realtime_overlay_frame(latest_frame)
        overlay_bytes = output.pop("overlay_png_bytes", b"")
        output["overlay_image_url"] = (
            "data:image/png;base64," + base64.b64encode(overlay_bytes).decode("utf-8") if overlay_bytes else ""
        )
        return output

    def run_realtime_overlay_bytes(self, frame_bytes: bytes) -> dict:
        latest_frame = self.decode_realtime_frame_bytes(frame_bytes)
        if latest_frame is None:
            raise ValueError("实时 pose 覆盖层没有收到有效视频帧。")
        return self._run_realtime_overlay_frame(latest_frame)

    @staticmethod
    def encode_overlay_ws_packet(data: dict) -> bytes:
        payload = data.get("overlay_png_bytes") or b""
        meta = {
            "has_pose": bool(data.get("has_pose")) and bool(payload),
            "message": str(data.get("message") or ""),
            "reused_previous_pose": bool(data.get("reused_previous_pose")),
            "tracking_mode": str(data.get("tracking_mode") or ""),
            "error": bool(data.get("error")),
        }
        meta_bytes = json.dumps(meta, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return struct.pack(">I", len(meta_bytes)) + meta_bytes + payload

    def run_realtime_inference(self, frames: list[str]) -> dict:
        decoded_frames = self.decode_realtime_frames(frames)
        pose_data, latest_frame, latest_keypoints, latest_scores = self.extract_pose_from_frames(
            decoded_frames,
            max_frames=self.realtime_profile.get("max_inference_frames", 32),
        )
        valid_pose_frames = self.count_valid_pose_frames(pose_data)
        if valid_pose_frames <= 0:
            return {
                "result": "",
                "overlay_image_url": "",
                "inference_seconds": 0.0,
                "has_pose": False,
                "message": "最近一段滑窗里没有检测到有效 pose，暂不发起字幕解码。请调整站位、退后一点并确保双手完整入镜。",
            }
        start_time = time.time()
        result = self.predict("", pose_data=pose_data)
        inference_seconds = round(time.time() - start_time, 2)
        overlay_image_url = ""
        if self._has_valid_pose(latest_scores):
            overlay_image_url = self.encode_overlay_image(latest_frame, latest_keypoints, latest_scores)
        return {
            "result": result,
            "overlay_image_url": overlay_image_url,
            "inference_seconds": inference_seconds,
            "has_pose": True,
            "message": f"实时推理完成，耗时约 {inference_seconds} 秒，本轮有效 pose 帧 {valid_pose_frames} 张。",
        }

    def run_sample_inference(self, sample: dict) -> dict:
        pose_path = sample.get("pose_path", "")
        if not pose_path or not os.path.exists(pose_path):
            raise ValueError("默认样本缺少预计算 pose 文件。")

        pose_data = load_pose_file(pose_path)
        pose_preview_url = self._get_or_create_sample_pose_preview(sample, pose_data)

        start_time = time.time()
        result = self.predict(sample["path"], pose_data=pose_data)
        inference_seconds = round(time.time() - start_time, 2)
        return {
            "result": result,
            "pose_video_url": pose_preview_url,
            "inference_seconds": inference_seconds,
        }

    def _get_or_create_sample_pose_preview(self, sample: dict, pose_data: dict) -> str:
        media_name = self.samples.cache_key(sample)
        file_path = self.results.root_dir / media_name
        if file_path.exists() and file_path.is_file() and file_path.stat().st_size > 0:
            return f"/media/{media_name}"

        pose_video_path = self.render_pose_video_from_pose(sample["path"], pose_data)
        try:
            shutil.copyfile(pose_video_path, file_path)
        finally:
            if os.path.exists(pose_video_path):
                os.remove(pose_video_path)
        return f"/media/{media_name}"

    def render_pose_video_from_pose(self, video_path: str, pose_data: dict) -> str:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            cap.release()
            raise ValueError("无法读取视频尺寸，无法生成默认样本 pose 可视化视频。")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            raw_output_path = tmp_file.name

        writer = cv2.VideoWriter(
            raw_output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError("无法创建默认样本 pose 可视化视频文件。")

        keypoints_seq = pose_data.get("keypoints") or []
        scores_seq = pose_data.get("scores") or []
        total_pose_frames = min(len(keypoints_seq), len(scores_seq))
        scale = np.array([width, height], dtype=np.float32)[None, None, :]

        try:
            frame_index = 0
            while frame_index < total_pose_frames:
                ok, frame = cap.read()
                if not ok:
                    break

                keypoints = np.asarray(keypoints_seq[frame_index], dtype=np.float32)
                scores = np.asarray(scores_seq[frame_index], dtype=np.float32)
                if keypoints.ndim == 2:
                    keypoints = keypoints[None, :, :]
                if scores.ndim == 1:
                    scores = scores[None, :]

                pose_frame = draw_skeleton(
                    frame.copy(),
                    keypoints * scale,
                    scores,
                    openpose_skeleton=False,
                    kpt_thr=0.3,
                    radius=2,
                    line_width=2,
                )
                writer.write(pose_frame)
                frame_index += 1
        finally:
            cap.release()
            writer.release()

        if os.path.getsize(raw_output_path) == 0:
            os.remove(raw_output_path)
            raise ValueError("默认样本 pose 可视化视频生成失败，输出文件为空。")

        return self._transcode_pose_video_for_browser(raw_output_path)

    def _get_pose_estimator(self):
        with self.pose_lock:
            if self.pose_estimator is None:
                self.pose_estimator = Wholebody(
                    to_openpose=False,
                    mode="lightweight",
                    backend="onnxruntime",
                    device=get_runtime_device(self.args),
                )
            return self.pose_estimator

    def _get_overlay_pose_estimator(self):
        with self.overlay_pose_lock:
            if self.overlay_pose_estimator is None:
                self.overlay_pose_estimator = Wholebody(
                    to_openpose=False,
                    mode="lightweight",
                    backend="onnxruntime",
                    device=get_runtime_device(self.args),
                )
            return self.overlay_pose_estimator

    def _estimate_pose(self, frame: np.ndarray):
        estimator = self._get_pose_estimator()
        with self.pose_infer_lock:
            return estimator(frame)

    def _estimate_overlay_pose(self, frame: np.ndarray):
        estimator = self._get_overlay_pose_estimator()
        with self.overlay_pose_infer_lock:
            return estimator(frame)

    def render_pose_video_and_pose(self, video_path: str) -> tuple[str, dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件：{video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if fps <= 0:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            raise ValueError("无法读取视频尺寸，无法生成 pose 可视化视频。")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            raw_output_path = tmp_file.name

        writer = cv2.VideoWriter(
            raw_output_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            cap.release()
            raise ValueError("无法创建 pose 可视化视频文件。")

        pose_data = {"keypoints": [], "scores": []}
        scale = np.array([width, height], dtype=np.float32)[None, None, :]

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                keypoints, scores = self._estimate_pose(frame)
                pose_data["keypoints"].append(np.asarray(keypoints, dtype=np.float32) / scale)
                pose_data["scores"].append(np.asarray(scores, dtype=np.float32))
                pose_frame = draw_skeleton(
                    frame.copy(),
                    keypoints,
                    scores,
                    openpose_skeleton=False,
                    kpt_thr=0.3,
                    radius=2,
                    line_width=2,
                )
                writer.write(pose_frame)
        finally:
            cap.release()
            writer.release()

        if os.path.getsize(raw_output_path) == 0:
            os.remove(raw_output_path)
            raise ValueError("pose 可视化视频生成失败，输出文件为空。")

        return self._transcode_pose_video_for_browser(raw_output_path), pose_data

    def _transcode_pose_video_for_browser(self, source_path: str) -> str:
        ffmpeg_exe = _find_ffmpeg_executable()
        if not ffmpeg_exe:
            return source_path

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            output_path = tmp_file.name

        cmd = [
            ffmpeg_exe,
            "-y",
            "-i",
            source_path,
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            output_path,
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if proc.returncode != 0 or not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
            if os.path.exists(output_path):
                os.remove(output_path)
            if os.path.exists(source_path):
                os.remove(source_path)
            raise ValueError(f"pose 可视化视频转码失败：{proc.stdout[-500:]}")

        os.remove(source_path)
        return output_path

    def run_inference_with_preview(self, video_path: str) -> dict:
        pose_video_path, pose_data = self.render_pose_video_and_pose(video_path)
        try:
            start_time = time.time()
            result = self.predict(video_path, pose_data=pose_data)
            inference_seconds = round(time.time() - start_time, 2)
            media_name = self.results.register_pose_video(pose_video_path)
            return {
                "result": result,
                "pose_video_url": f"/media/{media_name}",
                "inference_seconds": inference_seconds,
            }
        finally:
            if os.path.exists(pose_video_path):
                os.remove(pose_video_path)


class IPv6ThreadingHTTPServer(ThreadingHTTPServer):
    address_family = socket.AF_INET6

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
        except OSError:
            pass
        super().server_bind()


def make_handler(app: UniSignWebService):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            parsed = urlparse(self.path)
            if parsed.path == "/ws/realtime/overlay":
                self._handle_realtime_overlay_ws()
                return
            if parsed.path.startswith("/media/"):
                self._handle_media(parsed.path, head_only=False)
                return
            if parsed.path.startswith("/sample-media/"):
                self._handle_sample_media(parsed.path, head_only=False)
                return
            if parsed.path == "/api/job":
                self._handle_job_status(parsed)
                return
            if self.path not in {"/", "/index.html"}:
                self.send_error(404, "Not Found")
                return
            self._send_html(render_page(app_args=app.args, sample_videos=app.samples.list(), realtime_config=app.realtime_profile))

        def do_HEAD(self):
            parsed = urlparse(self.path)
            if parsed.path.startswith("/media/"):
                self._handle_media(parsed.path, head_only=True)
                return
            if parsed.path.startswith("/sample-media/"):
                self._handle_sample_media(parsed.path, head_only=True)
                return
            if parsed.path in {"/", "/index.html"}:
                body = render_page(app_args=app.args, sample_videos=app.samples.list(), realtime_config=app.realtime_profile).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.close_connection = True
                return
            self.send_error(404, "Not Found")

        def do_POST(self):
            parsed = urlparse(self.path)
            if parsed.path == "/api/upload/init":
                self._handle_upload_init()
                return
            if parsed.path == "/api/sample/start":
                self._handle_sample_start()
                return
            if parsed.path == "/api/realtime/overlay":
                self._handle_realtime_overlay()
                return
            if parsed.path == "/api/realtime/infer":
                self._handle_realtime_infer()
                return
            if parsed.path == "/api/upload/chunk":
                self._handle_upload_chunk(parsed)
                return
            if parsed.path == "/api/upload/complete":
                self._handle_upload_complete(parsed)
                return
            if parsed.path != "/predict":
                self.send_error(404, "Not Found")
                return

            content_length = int(self.headers.get("Content-Length", "0") or 0)
            if content_length > app.uploads.max_upload_bytes:
                self._send_html(
                    render_page(
                        error=(
                            f"上传文件过大：{content_length / (1024 * 1024):.1f} MB，"
                            f"当前服务上限为 {app.uploads.max_upload_bytes / (1024 * 1024):.1f} MB。"
                        ),
                        app_args=app.args,
                        sample_videos=app.samples.list(),
                        realtime_config=app.realtime_profile,
                    ),
                    status=413,
                )
                return

            temp_path = ""
            filename = ""
            try:
                form = cgi.FieldStorage(
                    fp=self.rfile,
                    headers=self.headers,
                    environ={
                        "REQUEST_METHOD": "POST",
                        "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    },
                )
                if "video" not in form:
                    raise ValueError("未收到视频文件，请重新上传。")

                video_field = form["video"]
                filename = os.path.basename(getattr(video_field, "filename", "") or "")
                if not filename:
                    raise ValueError("上传文件缺少文件名，请重新选择视频。")
                if not getattr(video_field, "file", None):
                    raise ValueError("上传内容为空，请重新选择视频。")

                suffix = Path(filename).suffix or ".mp4"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                    shutil.copyfileobj(video_field.file, temp_file)
                    temp_path = temp_file.name

                output = app.run_inference_with_preview(temp_path)
                self._send_html(
                    render_page(
                        result=output["result"],
                        filename=filename,
                        pose_video_url=output.get("pose_video_url", ""),
                        app_args=app.args,
                        sample_videos=app.samples.list(),
                        realtime_config=app.realtime_profile,
                    )
                )
            except UploadTooLargeError as exc:
                self._send_html(render_page(error=str(exc), filename=filename, app_args=app.args, sample_videos=app.samples.list(), realtime_config=app.realtime_profile), status=413)
            except Exception as exc:
                self._send_html(render_page(error=str(exc), filename=filename, app_args=app.args, sample_videos=app.samples.list(), realtime_config=app.realtime_profile), status=500)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.remove(temp_path)

        def log_message(self, fmt, *args):
            print("[web]", fmt % args)

        def _handle_upload_init(self):
            try:
                payload = self._read_json()
                filename = os.path.basename((payload.get("filename") or "").strip())
                if not filename:
                    raise ValueError("上传文件缺少文件名，请重新选择视频。")
                upload_id = app.uploads.start(filename=filename, total_size=int(payload.get("size") or 0))
                self._send_json({"upload_id": upload_id})
            except UploadTooLargeError as exc:
                self._send_json({"error": str(exc)}, status=413)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _handle_sample_start(self):
            try:
                payload = self._read_json(allow_empty=True)
                sample_id = str(payload.get("sample_id") or payload.get("sampleId") or "").strip()
                if not sample_id:
                    raise ValueError("缺少 sample_id。")
                sample = app.samples.get(sample_id)
                if sample is None:
                    raise ValueError("默认样本不存在，请刷新页面后重试。")
                self._send_json(
                    {
                        "job_id": app.jobs.start_sample(sample),
                        "filename": sample["filename"],
                        "status": "queued",
                        "message": "已开始处理默认样本。",
                    }
                )
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _handle_realtime_infer(self):
            try:
                payload = self._read_json()
                frames = payload.get("frames") or []
                if not isinstance(frames, list) or not frames:
                    raise ValueError("实时推理缺少有效 frames。")
                output = app.run_realtime_inference(frames)
                self._send_json(output)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _handle_realtime_overlay(self):
            try:
                content_type = (self.headers.get("Content-Type", "") or "").split(";", 1)[0].strip().lower()
                if content_type == "application/octet-stream":
                    content_length = int(self.headers.get("Content-Length", "0") or 0)
                    frame_bytes = self.rfile.read(content_length) if content_length > 0 else b""
                    if not frame_bytes:
                        raise ValueError("实时 pose 覆盖层缺少 frame。")
                    output = app.run_realtime_overlay_bytes(frame_bytes)
                    self._send_overlay(output)
                    return

                payload = self._read_json()
                frame_data = str(payload.get("frame") or "")
                if not frame_data:
                    raise ValueError("实时 pose 覆盖层缺少 frame。")
                output = app.run_realtime_overlay(frame_data)
                self._send_json(output)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _handle_realtime_overlay_ws(self):
            upgrade = (self.headers.get("Upgrade", "") or "").strip().lower()
            connection_header = (self.headers.get("Connection", "") or "").strip().lower()
            websocket_key = (self.headers.get("Sec-WebSocket-Key", "") or "").strip()
            websocket_version = (self.headers.get("Sec-WebSocket-Version", "") or "").strip()

            if upgrade != "websocket" or "upgrade" not in connection_header or not websocket_key:
                self.send_error(400, "Expected WebSocket upgrade")
                return
            if websocket_version and websocket_version != "13":
                self.send_response(426, "Upgrade Required")
                self.send_header("Sec-WebSocket-Version", "13")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True
                return

            accept_key = base64.b64encode(
                hashlib.sha1((websocket_key + WEBSOCKET_ACCEPT_GUID).encode("ascii")).digest()
            ).decode("ascii")

            self.send_response(101, "Switching Protocols")
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept_key)
            self.end_headers()
            self.wfile.flush()
            self.close_connection = True

            try:
                self.connection.settimeout(15.0)
                while True:
                    opcode, payload = self._read_websocket_message()
                    if opcode == 0x8:
                        break
                    if opcode != 0x2:
                        output = {
                            "overlay_png_bytes": b"",
                            "has_pose": False,
                            "reused_previous_pose": False,
                            "tracking_mode": "error",
                            "message": "实时 pose WebSocket 仅支持二进制 JPEG 帧。",
                            "error": True,
                        }
                    else:
                        try:
                            output = app.run_realtime_overlay_bytes(payload)
                        except Exception as exc:
                            output = {
                                "overlay_png_bytes": b"",
                                "has_pose": False,
                                "reused_previous_pose": False,
                                "tracking_mode": "error",
                                "message": str(exc),
                                "error": True,
                            }
                    self._send_websocket_frame(app.encode_overlay_ws_packet(output), opcode=0x2)
            except (BrokenPipeError, ConnectionError, ConnectionResetError, OSError, socket.timeout, TimeoutError):
                return
            except Exception:
                try:
                    error_output = {
                        "overlay_png_bytes": b"",
                        "has_pose": False,
                        "reused_previous_pose": False,
                        "tracking_mode": "error",
                        "message": "实时 pose WebSocket 连接已中断，请稍后自动重试。",
                        "error": True,
                    }
                    self._send_websocket_frame(app.encode_overlay_ws_packet(error_output), opcode=0x2)
                except Exception:
                    return
            finally:
                try:
                    self._send_websocket_close(1000)
                except Exception:
                    pass

        def _recv_ws_exactly(self, size: int) -> bytes:
            if size <= 0:
                return b""
            chunks = []
            remaining = size
            while remaining > 0:
                chunk = self.rfile.read(remaining)
                if not chunk:
                    raise ConnectionError("WebSocket connection closed while reading frame.")
                chunks.append(chunk)
                remaining -= len(chunk)
            return b"".join(chunks)

        def _read_websocket_message(self) -> tuple[int, bytes]:
            message_opcode = None
            fragments = []

            while True:
                header = self._recv_ws_exactly(2)
                first_byte, second_byte = header[0], header[1]
                fin = bool(first_byte & 0x80)
                opcode = first_byte & 0x0F
                masked = bool(second_byte & 0x80)
                payload_length = second_byte & 0x7F

                if payload_length == 126:
                    payload_length = struct.unpack(">H", self._recv_ws_exactly(2))[0]
                elif payload_length == 127:
                    payload_length = struct.unpack(">Q", self._recv_ws_exactly(8))[0]

                mask_key = self._recv_ws_exactly(4) if masked else b""
                payload = self._recv_ws_exactly(payload_length) if payload_length else b""
                if mask_key:
                    payload = bytes(byte ^ mask_key[index % 4] for index, byte in enumerate(payload))

                if opcode == 0x8:
                    return 0x8, payload
                if opcode == 0x9:
                    self._send_websocket_frame(payload, opcode=0xA)
                    continue
                if opcode == 0xA:
                    continue
                if opcode in (0x1, 0x2):
                    if message_opcode is not None:
                        raise ValueError("Unexpected fragmented WebSocket message.")
                    message_opcode = opcode
                    fragments.append(payload)
                    if fin:
                        return message_opcode, b"".join(fragments)
                    continue
                if opcode == 0x0:
                    if message_opcode is None:
                        raise ValueError("Unexpected WebSocket continuation frame.")
                    fragments.append(payload)
                    if fin:
                        return message_opcode, b"".join(fragments)
                    continue
                raise ValueError(f"Unsupported WebSocket opcode: {opcode}")

        def _send_websocket_frame(self, payload: bytes, opcode: int = 0x2, fin: bool = True):
            payload = payload or b""
            header = bytearray()
            header.append((0x80 if fin else 0x00) | (opcode & 0x0F))
            payload_length = len(payload)
            if payload_length < 126:
                header.append(payload_length)
            elif payload_length < (1 << 16):
                header.append(126)
                header.extend(struct.pack(">H", payload_length))
            else:
                header.append(127)
                header.extend(struct.pack(">Q", payload_length))
            self.connection.sendall(bytes(header) + payload)

        def _send_websocket_close(self, code: int = 1000, reason: str = ""):
            payload = struct.pack(">H", int(code))
            if reason:
                payload += reason.encode("utf-8")[:123]
            self._send_websocket_frame(payload, opcode=0x8)

        def _send_overlay(self, data: dict):
            payload = data.get("overlay_png_bytes") or b""
            has_pose = bool(data.get("has_pose")) and bool(payload)
            status = 200 if has_pose else 204
            self.send_response(status)
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Has-Pose", "1" if has_pose else "0")
            self.send_header("X-Pose-Message", quote(str(data.get("message") or ""), safe=""))
            self.send_header("X-Pose-Reused", "1" if data.get("reused_previous_pose") else "0")
            self.send_header("X-Pose-Tracking-Mode", str(data.get("tracking_mode") or ""))
            if has_pose:
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(payload)))
            else:
                self.send_header("Content-Length", "0")
            self.end_headers()
            if has_pose:
                self.wfile.write(payload)
                self.wfile.flush()
            self.close_connection = True

        def _handle_upload_chunk(self, parsed):
            upload_id = ""
            try:
                query = parse_qs(parsed.query)
                upload_id = self._extract_upload_id(query=query)
                if not upload_id:
                    raise ValueError("缺少 upload_id。")
                index = int((query.get("index") or [self.headers.get("X-Chunk-Index", "0")])[0])
                content_length = int(self.headers.get("Content-Length", "0") or 0)
                if content_length > app.uploads.max_chunk_bytes:
                    raise UploadTooLargeError(
                        f"单个分片过大：{content_length / (1024 * 1024):.1f} MB，当前分片上限为 {app.uploads.max_chunk_bytes / (1024 * 1024):.1f} MB。"
                    )
                chunk = self.rfile.read(content_length)
                info = app.uploads.append_chunk(upload_id=upload_id, index=index, chunk=chunk)
                self._send_json(info)
            except UploadTooLargeError as exc:
                self._send_json({"error": str(exc)}, status=413)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=400)

        def _handle_upload_complete(self, parsed):
            upload_id = ""
            try:
                payload = self._read_json(allow_empty=True)
                upload_id = self._extract_upload_id(payload=payload, query=parse_qs(parsed.query))
                if not upload_id:
                    raise ValueError("缺少 upload_id。")
                temp_path, filename = app.uploads.finish(upload_id)
                self._send_json(
                    {
                        "job_id": app.jobs.start(upload_id=upload_id, video_path=temp_path, filename=filename),
                        "upload_id": upload_id,
                        "filename": filename,
                        "status": "queued",
                        "message": "上传完成，已转入后台推理。",
                    }
                )
            except UploadTooLargeError as exc:
                self._send_json({"error": str(exc)}, status=413)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def _extract_upload_id(self, payload=None, query=None):
            payload = payload or {}
            query = query or {}
            header_id = self.headers.get("X-Upload-Id", "")
            query_id = (query.get("upload_id") or [""])[0]
            body_id = (payload.get("upload_id") or payload.get("uploadId") or "")
            return str(body_id or query_id or header_id).strip()

        def _handle_media(self, path: str, head_only: bool = False):
            media_name = path.split("/media/", 1)[-1]
            file_path = app.results.resolve(media_name)
            if file_path is None:
                self.send_error(404, "Not Found")
                return

            self._serve_video_file(file_path, head_only=head_only)

        def _handle_sample_media(self, path: str, head_only: bool = False):
            media_name = path.split("/sample-media/", 1)[-1]
            file_path = app.samples.resolve_media(media_name)
            if file_path is None:
                self.send_error(404, "Not Found")
                return

            self._serve_video_file(file_path, head_only=head_only)

        def _serve_video_file(self, file_path: Path, head_only: bool = False):
            if not file_path.exists() or not file_path.is_file():
                self.send_error(404, "Not Found")
                return

            file_size = file_path.stat().st_size
            range_header = self.headers.get("Range", "").strip()
            start = 0
            end = file_size - 1
            status = 200

            if range_header.startswith("bytes="):
                try:
                    start_text, end_text = range_header[6:].split("-", 1)
                    if start_text:
                        start = int(start_text)
                    if end_text:
                        end = int(end_text)
                    if start < 0 or end < start or end >= file_size:
                        raise ValueError
                    status = 206
                except ValueError:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.close_connection = True
                    return

            content_length = end - start + 1
            self.send_response(status)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Content-Length", str(content_length))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Accept-Ranges", "bytes")
            if status == 206:
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.end_headers()

            if head_only:
                self.close_connection = True
                return

            with open(file_path, "rb") as media_file:
                media_file.seek(start)
                remaining = content_length
                while remaining > 0:
                    chunk = media_file.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    remaining -= len(chunk)
            self.wfile.flush()
            self.close_connection = True

        def _handle_job_status(self, parsed):
            query = parse_qs(parsed.query)
            job_id = (query.get("job_id") or [""])[0].strip()
            upload_id = (query.get("upload_id") or [""])[0].strip()
            job = None
            if job_id:
                job = app.jobs.get(job_id)
            elif upload_id:
                job = app.jobs.get_by_upload_id(upload_id)
            else:
                job = app.jobs.get_latest()
            if job is None:
                self._send_json({"error": "任务不存在或已过期。"}, status=404)
                return
            self._send_json(job)

        def _read_json(self, allow_empty: bool = False):
            content_length = int(self.headers.get("Content-Length", "0") or 0)
            raw = self.rfile.read(content_length) if content_length > 0 else (b"{}" if not allow_empty else b"")
            if allow_empty and not raw.strip():
                return {}
            try:
                return json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError("请求不是合法的 JSON。") from exc

        def _send_html(self, body: str, status: int = 200):
            payload = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            self.close_connection = True

        def _send_json(self, data: dict, status: int = 200):
            payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Connection", "close")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            self.wfile.write(payload)
            self.wfile.flush()
            self.close_connection = True

    return Handler


def parse_args():
    parser = argparse.ArgumentParser("Uni-Sign web app", parents=[utils.get_args_parser()])
    parser.add_argument("--host", default="0.0.0.0", type=str, help="HTTP server host")
    parser.add_argument("--port", default=7860, type=int, help="HTTP server port")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device")
    parser.add_argument("--max-upload-size-mb", default=512, type=int, help="Maximum accepted video size in MB")
    parser.add_argument("--chunk-size-mb", default=0.5, type=float, help="Browser upload chunk size in MB")
    return parser.parse_args()


def main():
    args = parse_args()
    app = UniSignWebService(args)
    server_cls = IPv6ThreadingHTTPServer if ":" in args.host else ThreadingHTTPServer
    server = server_cls((args.host, args.port), make_handler(app))
    print(f"Uni-Sign web app listening on http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

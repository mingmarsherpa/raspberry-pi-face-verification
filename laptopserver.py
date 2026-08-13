import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Set

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, responses

from src.detect.sface import SFaceRecognizer
from src.detect.yunet import YunetDetector

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("video")

CAPTURES_DIR = "captures"
KNOWN_FACES_DIR = "known_faces"
AI_SAMPLE_INTERVAL_SECONDS = 0.2
CLEAR_ACTIVE_AFTER_NO_FACE_CHECKS = 3
FPS_LOG_WINDOW_SECONDS = 5.0

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)


class FrameBroadcaster:
    def __init__(self):
        self._subscribers: Set[asyncio.Queue] = set()

    def subscribe(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1)
        self._subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        self._subscribers.discard(q)

    def publish(self, frame: bytes) -> None:
        for q in list(self._subscribers):
            if q.full():
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            q.put_nowait(frame)


raw_broadcaster = FrameBroadcaster()
annotated_broadcaster = FrameBroadcaster()


def _process_sampled_frame(
    frame_bytes: bytes,
    detector: YunetDetector,
    recognizer: SFaceRecognizer,
    run_recognition: bool,
    saved_faces_cache: dict,
) -> tuple[bytes, bool, Optional[str], Optional[float]]:
    img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return frame_bytes, False, None, None

    raw_faces = detector.detect_raw(img)
    if len(raw_faces) == 0:
        return frame_bytes, False, None, None

    img_h, img_w, _ = img.shape
    for face in raw_faces:
        x, y, w, h = map(int, face[:4])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    recognized_name: Optional[str] = None
    recognized_score: Optional[float] = None

    if run_recognition:
        primary_face = raw_faces[0]
        recognized_name, recognized_score = recognizer.recognize(img, primary_face)
        x, y, w, h = map(int, primary_face[:4])
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(img_w, x + w), min(img_h, y + h)

        label = f"{recognized_name} ({recognized_score:.2f})"
        cv2.putText(
            img,
            label,
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        now_ts = time.time()
        if x2 > x1 and y2 > y1 and (now_ts - saved_faces_cache.get(recognized_name, 0.0) > 10.0):
            saved_faces_cache[recognized_name] = now_ts
            cropped = img[y1:y2, x1:x2]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_path = os.path.join(CAPTURES_DIR, f"{recognized_name}_{timestamp}.jpg")
            cv2.imwrite(saved_path, cropped)

    success, encoded = cv2.imencode(".jpg", img)
    if success:
        return encoded.tobytes(), True, recognized_name, recognized_score
    return frame_bytes, True, recognized_name, recognized_score


async def ai_worker():
    log.info("AI Worker starting...")
    queue = raw_broadcaster.subscribe()

    detector: Optional[YunetDetector] = None
    recognizer: Optional[SFaceRecognizer] = None
    try:
        detector = YunetDetector(conf_threshold=0.45)
        recognizer = SFaceRecognizer(known_faces_dir=KNOWN_FACES_DIR, detector=detector)
        log.info("AI models loaded (YuNet + SFace)")
    except Exception:
        log.exception("AI model init failed. Serving passthrough video only.")

    saved_faces_cache = {}
    face_active = False
    no_face_checks = 0
    next_sample_at = 0.0
    sampled_frame_count = 0
    sampled_window_start = time.monotonic()

    try:
        while True:
            frame_bytes = await queue.get()
            annotated_broadcaster.publish(frame_bytes)

            if detector is None or recognizer is None:
                continue

            now = time.monotonic()
            if now < next_sample_at:
                continue
            next_sample_at = now + AI_SAMPLE_INTERVAL_SECONDS

            sample_start = time.monotonic()

            annotated_bytes, has_face, recognized_name, recognized_score = await asyncio.to_thread(
                _process_sampled_frame,
                frame_bytes,
                detector,
                recognizer,
                not face_active,
                saved_faces_cache,
            )
            annotated_broadcaster.publish(annotated_bytes)

            sampled_frame_count += 1
            sampled_elapsed = time.monotonic() - sampled_window_start
            if sampled_elapsed >= FPS_LOG_WINDOW_SECONDS:
                log.info("AI sampled FPS: %.1f", sampled_frame_count / sampled_elapsed)
                sampled_frame_count = 0
                sampled_window_start = time.monotonic()

            if has_face:
                no_face_checks = 0
                if not face_active:
                    face_active = True
                    log.info("Face detected")
                    if recognized_name is not None and recognized_score is not None:
                        log.info("Face recognized: %s (score=%.3f)", recognized_name, recognized_score)
            elif face_active:
                no_face_checks += 1
                if no_face_checks >= CLEAR_ACTIVE_AFTER_NO_FACE_CHECKS:
                    face_active = False
                    no_face_checks = 0
                    log.info(
                        "Face state cleared after %d no-face checks",
                        CLEAR_ACTIVE_AFTER_NO_FACE_CHECKS,
                    )

            process_ms = (time.monotonic() - sample_start) * 1000.0
            log.info("AI sample processed in %.1f ms", process_ms)

    except asyncio.CancelledError:
        log.info("AI Worker stopped")
    finally:
        raw_broadcaster.unsubscribe(queue)


@asynccontextmanager
async def lifespan(app: FastAPI):
    ai_task = asyncio.create_task(ai_worker())
    yield
    ai_task.cancel()
    await asyncio.gather(ai_task, return_exceptions=True)


app = FastAPI(lifespan=lifespan)


@app.websocket("/ws/video")
async def ingest_video(websocket: WebSocket):
    await websocket.accept()
    log.info("Pi connected — ingesting video")
    frame_count = 0
    window_start = time.monotonic()
    try:
        while True:
            frame = await websocket.receive_bytes()
            raw_broadcaster.publish(frame)
            frame_count += 1
            elapsed = time.monotonic() - window_start
            if elapsed >= FPS_LOG_WINDOW_SECONDS:
                log.info("Camera ingest FPS: %.1f", frame_count / elapsed)
                frame_count = 0
                window_start = time.monotonic()
    except WebSocketDisconnect:
        log.info("Pi disconnected")


@app.websocket("/ws/view")
async def view_video(websocket: WebSocket):
    await websocket.accept()
    queue = annotated_broadcaster.subscribe()
    try:
        while True:
            frame = await queue.get()
            await websocket.send_bytes(frame)
    except WebSocketDisconnect:
        pass
    finally:
        annotated_broadcaster.unsubscribe(queue)


@app.get("/")
async def index():
    return responses.HTMLResponse("""
<!DOCTYPE html>
<html>
<head>
    <title>Live Stream</title>
</head>
<body style="background:#222; text-align:center; color:white; font-family:sans-serif;">
    <h2>Live Feed</h2>
    <div id="fps" style="font-size:18px; margin-bottom:10px; color:#00ff00;">FPS: 0</div>
    <img id="v" style="width:640px; border:2px solid #555;">
    <script>
        const img = document.getElementById("v");
        const fpsElem = document.getElementById("fps");
        const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws/view");
        ws.binaryType = "arraybuffer";

        let url;
        let frameCount = 0;
        let lastTime = performance.now();

        ws.onmessage = (e) => {
            if (url) URL.revokeObjectURL(url);
            url = URL.createObjectURL(new Blob([e.data], {type:"image/jpeg"}));
            img.src = url;

            frameCount++;
            const now = performance.now();
            if (now - lastTime >= 1000) {
                const fps = ((frameCount * 1000) / (now - lastTime)).toFixed(1);
                fpsElem.innerText = `FPS: ${fps}`;
                frameCount = 0;
                lastTime = now;
            }
        };
    </script>
</body>
</html>
""")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
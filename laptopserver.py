import asyncio
import logging
import os
import time
import random
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Set

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, responses

from src.detect.sface import SFaceRecognizer
from src.detect.yunet import YunetDetector

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("video")

CAPTURES_DIR = "captures"
KNOWN_FACES_DIR = "known_faces"
SUSPICIOUS_DIR = "suspicious"

AI_SAMPLE_INTERVAL_SECONDS = 0.2
CLEAR_ACTIVE_AFTER_NO_FACE_CHECKS = 3
FPS_LOG_WINDOW_SECONDS = 5.0
RECOGNITION_CONFIRMATION_SAMPLES = 3
LIVENESS_BASELINE_SAMPLES = 3
LIVENESS_REQUIRED_MOVEMENT_SAMPLES = 2
LIVENESS_TIMEOUT_SECONDS = 7.0
LIVENESS_YAW_DELTA_THRESHOLD = 0.16
LIVENESS_PITCH_DELTA_THRESHOLD = 0.12

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(SUSPICIOUS_DIR, exist_ok=True)

@dataclass(frozen=True)
class HeadPoseMeasurement:
    """Scale-independent pose proxies derived from YuNet's five landmarks."""

    yaw: float
    pitch: float


def measure_head_pose(primary_face: np.ndarray) -> Optional[HeadPoseMeasurement]:
    """Return normalized yaw and pitch proxies, or None for degenerate landmarks.

    YuNet landmarks are right eye, left eye, nose, right mouth corner, and left
    mouth corner. Comparing each frame with a neutral baseline avoids relying on
    a fixed facial geometry ratio.
    """
    right_eye = np.array(primary_face[4:6], dtype=float)
    left_eye = np.array(primary_face[6:8], dtype=float)
    nose = np.array(primary_face[8:10], dtype=float)
    right_mouth = np.array(primary_face[10:12], dtype=float)
    left_mouth = np.array(primary_face[12:14], dtype=float)

    eye_midpoint = (right_eye + left_eye) / 2.0
    mouth_midpoint = (right_mouth + left_mouth) / 2.0
    eye_distance = float(np.linalg.norm(right_eye - left_eye))
    eye_to_mouth_distance = float(mouth_midpoint[1] - eye_midpoint[1])
    if eye_distance < 1e-6 or abs(eye_to_mouth_distance) < 1e-6:
        return None

    return HeadPoseMeasurement(
        yaw=float((nose[0] - eye_midpoint[0]) / eye_distance),
        pitch=float((nose[1] - eye_midpoint[1]) / eye_to_mouth_distance),
    )


class ActiveLivenessManager:
    """Runs one randomized, baseline-relative head-movement challenge."""

    CHALLENGES = ("look left", "look right", "look up", "look down")

    def __init__(
        self,
        timeout_seconds: float = LIVENESS_TIMEOUT_SECONDS,
        baseline_samples: int = LIVENESS_BASELINE_SAMPLES,
        required_movement_samples: int = LIVENESS_REQUIRED_MOVEMENT_SAMPLES,
    ):
        self.timeout_seconds = timeout_seconds
        self.baseline_samples_required = baseline_samples
        self.required_movement_samples = required_movement_samples
        self.reset(log_reset=False)

    def start_new_challenge(self) -> str:
        self.reset(log_reset=False)
        self.current_challenge = random.choice(self.CHALLENGES)
        self.is_preparing = True
        log.info("LIVENESS: Hold still while a neutral pose is measured")
        return self.current_challenge

    def evaluate(self, pose: HeadPoseMeasurement, now: Optional[float] = None) -> bool:
        """Advance the challenge and return True only when it has passed."""
        now = time.monotonic() if now is None else now
        if self.is_passed or self.current_challenge is None:
            return self.is_passed

        if self.is_preparing:
            self._baseline_measurements.append(pose)
            if len(self._baseline_measurements) < self.baseline_samples_required:
                return False
            self._baseline = HeadPoseMeasurement(
                yaw=float(np.mean([sample.yaw for sample in self._baseline_measurements])),
                pitch=float(np.mean([sample.pitch for sample in self._baseline_measurements])),
            )
            self.is_preparing = False
            self.is_active = True
            self.challenge_start_time = now
            log.info("LIVENESS CHALLENGE: Please %s", self.current_challenge.upper())
            return False

        if not self.is_active:
            return False
        if now - self.challenge_start_time > self.timeout_seconds:
            log.warning("LIVENESS CHALLENGE: FAILED (timed out)")
            self.is_active = False
            self.is_failed = True
            return False

        if self._movement_matches_challenge(pose):
            self._matching_samples += 1
        else:
            self._matching_samples = 0

        if self._matching_samples >= self.required_movement_samples:
            self.is_passed = True
            self.is_active = False
            log.info("LIVENESS CHALLENGE: SUCCESS (%s verified)", self.current_challenge)
        return self.is_passed

    def _movement_matches_challenge(self, pose: HeadPoseMeasurement) -> bool:
        if self._baseline is None:
            return False
        yaw_delta = pose.yaw - self._baseline.yaw
        pitch_delta = pose.pitch - self._baseline.pitch
        if self.current_challenge == "look left":
            return yaw_delta >= LIVENESS_YAW_DELTA_THRESHOLD
        if self.current_challenge == "look right":
            return yaw_delta <= -LIVENESS_YAW_DELTA_THRESHOLD
        if self.current_challenge == "look up":
            return pitch_delta <= -LIVENESS_PITCH_DELTA_THRESHOLD
        return pitch_delta >= LIVENESS_PITCH_DELTA_THRESHOLD  # look down

    def reset(self, log_reset: bool = True) -> None:
        had_session = getattr(self, "current_challenge", None) is not None
        if log_reset and had_session:
            log.info("LIVENESS STATE: Reset (face left the frame)")
        self.is_active = False
        self.is_preparing = False
        self.is_passed = False
        self.is_failed = False
        self.current_challenge: Optional[str] = None
        self.challenge_start_time = 0.0
        self._baseline_measurements: list[HeadPoseMeasurement] = []
        self._baseline: Optional[HeadPoseMeasurement] = None
        self._matching_samples = 0

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
    spoof_detector: Any,  # NEW
    mask_detector: Any,   # NEW
    run_recognition: bool,
    saved_faces_cache: dict,
) -> tuple[bytes, bool, str, Optional[str], Optional[float], Optional[np.ndarray]]:
    
    img = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return frame_bytes, False, "none", None, None, None

    raw_faces = detector.detect_raw(img)
    if len(raw_faces) == 0:
        return frame_bytes, False, "none", None, None, None

    img_h, img_w, _ = img.shape
    primary_face = raw_faces[0]
    x, y, w, h = map(int, primary_face[:4])
    bbox = (x, y, w, h)
    
    # 1. Mask Check
    if mask_detector.is_masked(img, bbox):
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 3)
        cv2.putText(img, "FACE COVERED! REMOVE MASK", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        success, encoded = cv2.imencode(".jpg", img)
        return encoded.tobytes() if success else frame_bytes, True, "masked", None, None, primary_face

    # 2. Anti-Spoof Check
    is_real, spoof_score = spoof_detector.is_real(img, bbox)
    if not is_real:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 165, 255), 3)
        cv2.putText(img, "SPOOF DETECTED!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
        success, encoded = cv2.imencode(".jpg", img)
        return encoded.tobytes() if success else frame_bytes, True, "spoof", None, None, primary_face

    # 3. Standard Recognition (Only if clear)
    recognized_name, recognized_score = None, None
    if run_recognition:
        recognized_name, recognized_score = recognizer.recognize(img, primary_face)
        label = f"{recognized_name} ({recognized_score:.2f})"
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(img, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Cache saving logic for known faces
        now_ts = time.time()
        if now_ts - saved_faces_cache.get(recognized_name, 0.0) > 10.0 and recognized_name != "unknown":
            saved_faces_cache[recognized_name] = now_ts
            x1, y1 = max(0, x), max(0, y)
            x2, y2 = min(img_w, x + w), min(img_h, y + h)
            cropped = img[y1:y2, x1:x2]
            if cropped.size > 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                saved_path = os.path.join(CAPTURES_DIR, f"{recognized_name}_{timestamp}.jpg")
                cv2.imwrite(saved_path, cropped)

    success, encoded = cv2.imencode(".jpg", img)
    return encoded.tobytes() if success else frame_bytes, True, "clear", recognized_name, recognized_score, primary_face


async def ai_worker():
    log.info("AI Worker starting...")
    queue = raw_broadcaster.subscribe()
    #importing here to avoid circular import issues
    from src.detect.antispoof import AntiSpoofDetector
    from src.detect.mask import MaskDetector

    detector: Optional[YunetDetector] = None
    recognizer: Optional[SFaceRecognizer] = None
    spoof_detector: Optional[AntiSpoofDetector] = None
    mask_detector: Optional[MaskDetector] = None
    try:
        detector = YunetDetector(conf_threshold=0.45)
        recognizer = SFaceRecognizer(known_faces_dir=KNOWN_FACES_DIR, detector=detector)
        spoof_detector = AntiSpoofDetector()
        mask_detector = MaskDetector()
        log.info("AI models loaded (YuNet + SFace + AntiSpoof + Mask)")
    except Exception:
        log.exception("AI model init failed. Serving passthrough video only.")

    saved_faces_cache = {}
    suspicious_cache = 0.0
    face_active = False
    no_face_checks = 0
    confirmed_name: Optional[str] = None
    recognition_samples = 0
    liveness = ActiveLivenessManager()
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

            annotated_bytes, has_face, status, recognized_name, recognized_score, primary_face = await asyncio.to_thread(
                _process_sampled_frame,
                frame_bytes,
                detector,
                recognizer,
                spoof_detector,
                mask_detector,
                liveness.current_challenge is None,
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
                if status in ["masked", "spoof"]:
                    liveness.reset(log_reset=False)
                    confirmed_name = None
                    recognition_samples = 0
                    
                    now_ts = time.time()
                    # Throttle saving suspicious images to once every 5 seconds
                    if now_ts - suspicious_cache > 5.0:
                        suspicious_cache = now_ts
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filepath = os.path.join(SUSPICIOUS_DIR, f"{status}_{timestamp}.jpg")
                        with open(filepath, "wb") as f:
                            f.write(frame_bytes) # Save original unannotated frame
                        log.warning(f"Suspicious activity ({status}) recorded at {filepath}")
                    continue

                if liveness.current_challenge is None:
                    if recognized_name is not None and recognized_name != "unknown":
                        if recognized_name == confirmed_name:
                            recognition_samples += 1
                        else:
                            confirmed_name = recognized_name
                            recognition_samples = 1
                        if recognition_samples == RECOGNITION_CONFIRMATION_SAMPLES:
                            log.info(
                                "Face recognized: %s (score=%.3f)",
                                recognized_name,
                                recognized_score,
                            )
                            liveness.start_new_challenge()
                    else:
                        confirmed_name = None
                        recognition_samples = 0
                elif primary_face is not None:
                    pose = measure_head_pose(primary_face)
                    if pose is not None:
                        liveness.evaluate(pose)
                    if liveness.is_failed:
                        # Retry only after fresh, consistent recognition.
                        liveness.reset(log_reset=False)
                        confirmed_name = None
                        recognition_samples = 0
            elif face_active:
                no_face_checks += 1
                if no_face_checks >= CLEAR_ACTIVE_AFTER_NO_FACE_CHECKS:
                    face_active = False
                    no_face_checks = 0
                    confirmed_name = None
                    recognition_samples = 0
                    liveness.reset()
                    log.info(
                        "Face state cleared after %d no-face checks",
                        CLEAR_ACTIVE_AFTER_NO_FACE_CHECKS,
                    )

            process_ms = (time.monotonic() - sample_start) * 1000.0
           #log.info("AI sample processed in %.1f ms", process_ms)

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

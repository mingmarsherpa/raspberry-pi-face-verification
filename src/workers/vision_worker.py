import os
import time
import random
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

import cv2
import numpy as np

from src.core.bus import ZmqPublisher, ZmqSubscriber

from src.detect.sface import SFaceRecognizer
from src.detect.yunet import YunetDetector
from src.detect.antispoof import AntiSpoofDetector
from src.detect.mask import MaskDetector


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("vision_worker")


# ============================================================
# DIRECTORIES
# ============================================================

CAPTURES_DIR = "captures"
KNOWN_FACES_DIR = "known_faces"
SUSPICIOUS_DIR = "suspicious"

os.makedirs(CAPTURES_DIR, exist_ok=True)
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(SUSPICIOUS_DIR, exist_ok=True)


# ============================================================
# PERFORMANCE / STATE SETTINGS
# ============================================================

# AI runs approximately every 0.2 seconds = 5 FPS maximum.
# Your actual FPS may be around 4 FPS depending on model speed.
AI_SAMPLE_INTERVAL_SECONDS = 0.2

# Require this many consecutive samples without a face
# before considering the visitor gone.
CLEAR_ACTIVE_AFTER_NO_FACE_CHECKS = 3

FPS_LOG_WINDOW_SECONDS = 5.0


# ============================================================
# FACE RECOGNITION
# ============================================================

# Require recognition on 3 samples before accepting identity.
RECOGNITION_CONFIRMATION_SAMPLES = 3


# ============================================================
# LIVENESS
# ============================================================

LIVENESS_BASELINE_SAMPLES = 3
LIVENESS_REQUIRED_MOVEMENT_SAMPLES = 2

# User has 7 seconds to complete challenge.
LIVENESS_TIMEOUT_SECONDS = 7.0

# Head movement thresholds.
LIVENESS_YAW_DELTA_THRESHOLD = 0.16
LIVENESS_PITCH_DELTA_THRESHOLD = 0.12


# ============================================================
# ACCESS CONTROL
# ============================================================

# After success/failure, don't immediately start another session.
ACCESS_COOLDOWN_SECONDS = 3.0


# ============================================================
# HEAD POSE
# ============================================================

@dataclass(frozen=True)
class HeadPoseMeasurement:
    yaw: float
    pitch: float


def measure_head_pose(
    primary_face: np.ndarray,
) -> Optional[HeadPoseMeasurement]:

    right_eye = np.array(primary_face[4:6], dtype=float)
    left_eye = np.array(primary_face[6:8], dtype=float)

    nose = np.array(primary_face[8:10], dtype=float)

    right_mouth = np.array(primary_face[10:12], dtype=float)
    left_mouth = np.array(primary_face[12:14], dtype=float)

    eye_midpoint = (right_eye + left_eye) / 2.0
    mouth_midpoint = (right_mouth + left_mouth) / 2.0

    eye_distance = float(
        np.linalg.norm(right_eye - left_eye)
    )

    eye_to_mouth_distance = float(
        mouth_midpoint[1] - eye_midpoint[1]
    )

    if eye_distance < 1e-6:
        return None

    if abs(eye_to_mouth_distance) < 1e-6:
        return None

    return HeadPoseMeasurement(
        yaw=float(
            (nose[0] - eye_midpoint[0])
            / eye_distance
        ),
        pitch=float(
            (nose[1] - eye_midpoint[1])
            / eye_to_mouth_distance
        ),
    )


# ============================================================
# ACTIVE LIVENESS MANAGER
# ============================================================

class ActiveLivenessManager:

    CHALLENGES = (
        "look left",
        "look right",
        "look up",
        "look down",
    )

    def __init__(
        self,
        timeout_seconds=LIVENESS_TIMEOUT_SECONDS,
        baseline_samples=LIVENESS_BASELINE_SAMPLES,
        required_movement_samples=LIVENESS_REQUIRED_MOVEMENT_SAMPLES,
    ):
        self.timeout_seconds = timeout_seconds
        self.baseline_samples_required = baseline_samples
        self.required_movement_samples = required_movement_samples

        self.reset(log_reset=False)

    def start_new_challenge(self) -> str:

        self.reset(log_reset=False)

        self.current_challenge = random.choice(
            self.CHALLENGES
        )

        self.is_preparing = True

        log.info(
            "LIVENESS: Hold still while a neutral pose is measured"
        )

        return self.current_challenge

    def evaluate(
        self,
        pose: HeadPoseMeasurement,
        now: Optional[float] = None,
    ) -> bool:

        now = (
            time.monotonic()
            if now is None
            else now
        )

        if self.is_passed:
            return True

        if self.is_failed:
            return False

        if self.current_challenge is None:
            return False

        # ----------------------------------------------------
        # BASELINE
        # ----------------------------------------------------

        if self.is_preparing:

            self._baseline_measurements.append(pose)

            if (
                len(self._baseline_measurements)
                < self.baseline_samples_required
            ):
                return False

            self._baseline = HeadPoseMeasurement(
                yaw=float(
                    np.mean(
                        [
                            sample.yaw
                            for sample in self._baseline_measurements
                        ]
                    )
                ),
                pitch=float(
                    np.mean(
                        [
                            sample.pitch
                            for sample in self._baseline_measurements
                        ]
                    )
                ),
            )

            self.is_preparing = False
            self.is_active = True
            self.challenge_start_time = now

            log.info(
                "LIVENESS CHALLENGE: Please %s",
                self.current_challenge.upper(),
            )

            return False

        # ----------------------------------------------------
        # CHALLENGE ACTIVE
        # ----------------------------------------------------

        if not self.is_active:
            return False

        # ----------------------------------------------------
        # TIMEOUT
        # ----------------------------------------------------

        if (
            now - self.challenge_start_time
            > self.timeout_seconds
        ):

            log.warning(
                "LIVENESS CHALLENGE: FAILED (timed out)"
            )

            self.is_active = False
            self.is_failed = True

            return False

        # ----------------------------------------------------
        # CHECK MOVEMENT
        # ----------------------------------------------------

        if self._movement_matches_challenge(pose):

            self._matching_samples += 1

        else:

            self._matching_samples = 0

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        if (
            self._matching_samples
            >= self.required_movement_samples
        ):

            self.is_passed = True
            self.is_active = False

            log.info(
                "LIVENESS CHALLENGE: SUCCESS (%s verified)",
                self.current_challenge,
            )

        return self.is_passed

    def _movement_matches_challenge(
        self,
        pose: HeadPoseMeasurement,
    ) -> bool:

        if self._baseline is None:
            return False

        yaw_delta = (
            pose.yaw - self._baseline.yaw
        )

        pitch_delta = (
            pose.pitch - self._baseline.pitch
        )

        if self.current_challenge == "look left":

            return (
                yaw_delta
                >= LIVENESS_YAW_DELTA_THRESHOLD
            )

        if self.current_challenge == "look right":

            return (
                yaw_delta
                <= -LIVENESS_YAW_DELTA_THRESHOLD
            )

        if self.current_challenge == "look up":

            return (
                pitch_delta
                <= -LIVENESS_PITCH_DELTA_THRESHOLD
            )

        # look down

        return (
            pitch_delta
            >= LIVENESS_PITCH_DELTA_THRESHOLD
        )

    def reset(
        self,
        log_reset: bool = True,
    ) -> None:

        had_session = (
            getattr(
                self,
                "current_challenge",
                None,
            )
            is not None
        )

        if log_reset and had_session:

            log.info(
                "LIVENESS STATE: Reset (face left the frame)"
            )

        self.is_active = False
        self.is_preparing = False
        self.is_passed = False
        self.is_failed = False

        self.current_challenge: Optional[str] = None

        self.challenge_start_time = 0.0

        self._baseline_measurements = []

        self._baseline: Optional[
            HeadPoseMeasurement
        ] = None

        self._matching_samples = 0


# ============================================================
# FRAME PROCESSING
# ============================================================

def _process_sampled_frame(
    frame_bytes: bytes,
    detector: YunetDetector,
    recognizer: SFaceRecognizer,
    spoof_detector: Any,
    mask_detector: Any,
    run_recognition: bool,
    saved_faces_cache: dict,
):

    img = cv2.imdecode(
        np.frombuffer(frame_bytes, np.uint8),
        cv2.IMREAD_COLOR,
    )

    if img is None:

        return (
            frame_bytes,
            False,
            "none",
            None,
            None,
            None,
        )

    raw_faces = detector.detect_raw(img)

    if len(raw_faces) == 0:

        return (
            frame_bytes,
            False,
            "none",
            None,
            None,
            None,
        )

    img_h, img_w, _ = img.shape

    # For now we use the largest/primary detected face.
    primary_face = raw_faces[0]

    x, y, w, h = map(
        int,
        primary_face[:4],
    )

    bbox = (
        x,
        y,
        w,
        h,
    )

    # ========================================================
    # MASK DETECTION
    # ========================================================

    if mask_detector.is_masked(
        img,
        bbox,
    ):

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            img,
            "FACE COVERED! REMOVE MASK",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        success, encoded = cv2.imencode(
            ".jpg",
            img,
        )

        return (
            encoded.tobytes()
            if success
            else frame_bytes,
            True,
            "masked",
            None,
            None,
            primary_face,
        )

    # ========================================================
    # ANTI-SPOOF
    # ========================================================

    is_real, spoof_score = (
        spoof_detector.is_real(
            img,
            bbox,
        )
    )

    if not is_real:

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 165, 255),
            3,
        )

        cv2.putText(
            img,
            "SPOOF DETECTED!",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 165, 255),
            2,
        )

        success, encoded = cv2.imencode(
            ".jpg",
            img,
        )

        return (
            encoded.tobytes()
            if success
            else frame_bytes,
            True,
            "spoof",
            None,
            None,
            primary_face,
        )

    # ========================================================
    # FACE RECOGNITION
    # ========================================================

    recognized_name = None
    recognized_score = None

    if run_recognition:

        (
            recognized_name,
            recognized_score,
        ) = recognizer.recognize(
            img,
            primary_face,
        )

        label = (
            f"{recognized_name} "
            f"({recognized_score:.2f})"
        )

        cv2.rectangle(
            img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            img,
            label,
            (
                x,
                max(
                    20,
                    y - 10,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        # ----------------------------------------------------
        # SAVE OCCASIONAL CAPTURE
        # ----------------------------------------------------

        now_ts = time.time()

        if (
            recognized_name != "unknown"
            and recognized_name is not None
            and (
                now_ts
                - saved_faces_cache.get(
                    recognized_name,
                    0.0,
                )
                > 10.0
            )
        ):

            saved_faces_cache[
                recognized_name
            ] = now_ts

            x1 = max(0, x)
            y1 = max(0, y)

            x2 = min(
                img_w,
                x + w,
            )

            y2 = min(
                img_h,
                y + h,
            )

            cropped = img[
                y1:y2,
                x1:x2,
            ]

            if cropped.size > 0:

                timestamp = (
                    datetime.now()
                    .strftime(
                        "%Y%m%d_%H%M%S"
                    )
                )

                saved_path = os.path.join(
                    CAPTURES_DIR,
                    f"{recognized_name}_{timestamp}.jpg",
                )

                cv2.imwrite(
                    saved_path,
                    cropped,
                )

    # ========================================================
    # ENCODE
    # ========================================================

    success, encoded = cv2.imencode(
        ".jpg",
        img,
    )

    return (
        encoded.tobytes()
        if success
        else frame_bytes,
        True,
        "clear",
        recognized_name,
        recognized_score,
        primary_face,
    )


# ============================================================
# MAIN VISION WORKER
# ============================================================

def run_vision_worker():

    log.info(
        "Vision Worker starting..."
    )

    # ========================================================
    # ZMQ
    # ========================================================

    frame_sub = ZmqSubscriber(
        "camera.frames",
        conflate=True,
    )

    anno_pub = ZmqPublisher(
        "camera.annotated"
    )

    event_pub = ZmqPublisher(
        "vision.events"
    )

    # ========================================================
    # LOAD MODELS
    # ========================================================

    try:

        detector = YunetDetector(
            conf_threshold=0.45
        )

        recognizer = SFaceRecognizer(
            known_faces_dir=KNOWN_FACES_DIR,
            detector=detector,
        )

        spoof_detector = AntiSpoofDetector()

        mask_detector = MaskDetector()

        log.info(
            "AI models loaded "
            "(YuNet + SFace + AntiSpoof + Mask)"
        )

    except Exception as e:

        log.exception(
            "AI model init failed: %s",
            e,
        )

        return

    # ========================================================
    # STATE
    # ========================================================

    saved_faces_cache = {}

    suspicious_cache = 0.0

    face_active = False

    no_face_checks = 0

    confirmed_name: Optional[str] = None

    recognition_samples = 0

    liveness = ActiveLivenessManager()

    # --------------------------------------------------------
    # IMPORTANT:
    # Prevent repeated access_granted events.
    # --------------------------------------------------------

    access_granted_sent = False

    # --------------------------------------------------------
    # Cooldown after successful/failed authentication.
    # --------------------------------------------------------

    cooldown_until = 0.0

    # ========================================================
    # FPS
    # ========================================================

    next_sample_at = 0.0

    sampled_frame_count = 0

    sampled_window_start = (
        time.monotonic()
    )

    log.info(
        "Entering main vision loop..."
    )

    # ========================================================
    # MAIN LOOP
    # ========================================================

    while True:

        try:

            # ------------------------------------------------
            # RECEIVE FRAME
            # ------------------------------------------------

            frame_bytes = (
                frame_sub.receive_bytes()
            )

            now = time.monotonic()

            # ------------------------------------------------
            # AI SAMPLING
            # ------------------------------------------------

            if now < next_sample_at:

                # Don't run AI on every frame.
                # Pass frame directly to viewer.
                anno_pub.publish_bytes(
                    frame_bytes
                )

                continue

            next_sample_at = (
                now
                + AI_SAMPLE_INTERVAL_SECONDS
            )

            # ------------------------------------------------
            # CHECK COOLDOWN
            # ------------------------------------------------

            in_cooldown = (
                now < cooldown_until
            )

            # During cooldown we still process the frame
            # for display, but don't start a new auth session.
            run_recognition = (
                liveness.current_challenge is None
                and not in_cooldown
            )

            # ------------------------------------------------
            # PROCESS FRAME
            # ------------------------------------------------

            (
                annotated_bytes,
                has_face,
                status,
                recognized_name,
                recognized_score,
                primary_face,
            ) = _process_sampled_frame(
                frame_bytes,
                detector,
                recognizer,
                spoof_detector,
                mask_detector,
                run_recognition,
                saved_faces_cache,
            )

            # ------------------------------------------------
            # PUBLISH ANNOTATED FRAME
            # ------------------------------------------------

            anno_pub.publish_bytes(
                annotated_bytes
            )

            # ------------------------------------------------
            # FPS LOGGING
            # ------------------------------------------------

            sampled_frame_count += 1

            if (
                now - sampled_window_start
                >= FPS_LOG_WINDOW_SECONDS
            ):

                fps = (
                    sampled_frame_count
                    / (
                        now
                        - sampled_window_start
                    )
                )

                log.info(
                    "AI sampled FPS: %.1f",
                    fps,
                )

                sampled_frame_count = 0

                sampled_window_start = now

            # =================================================
            # FACE PRESENT
            # =================================================

            if has_face:

                no_face_checks = 0

                # ------------------------------------------------
                # FIRST FACE DETECTION
                # ------------------------------------------------

                if not face_active:

                    face_active = True

                    log.info(
                        "Face detected"
                    )

                    event_pub.publish_json(
                        {
                            "event": "face_detected"
                        }
                    )

                # =================================================
                # MASK / SPOOF
                # =================================================

                if status in (
                    "masked",
                    "spoof",
                ):

                    # Cancel current authentication.
                    liveness.reset(
                        log_reset=False
                    )

                    confirmed_name = None
                    recognition_samples = 0

                    access_granted_sent = False

                    now_ts = time.time()

                    # Avoid saving the same suspicious event
                    # continuously.
                    if (
                        now_ts
                        - suspicious_cache
                        > 5.0
                    ):

                        suspicious_cache = now_ts

                        filepath = os.path.join(
                            SUSPICIOUS_DIR,
                            f"{status}_{int(now_ts)}.jpg",
                        )

                        with open(
                            filepath,
                            "wb",
                        ) as f:

                            f.write(
                                frame_bytes
                            )

                        log.warning(
                            "Suspicious activity (%s) recorded",
                            status,
                        )

                        event_pub.publish_json(
                            {
                                "event": "threat_detected",
                                "type": status,
                            }
                        )

                    continue

                # =================================================
                # COOLDOWN
                # =================================================

                if in_cooldown:

                    continue

                # =================================================
                # BEFORE LIVENESS CHALLENGE
                # =================================================

                if (
                    liveness.current_challenge
                    is None
                ):

                    if (
                        recognized_name is not None
                        and recognized_name != "unknown"
                    ):

                        # -----------------------------------------
                        # SAME PERSON
                        # -----------------------------------------

                        if (
                            recognized_name
                            == confirmed_name
                        ):

                            recognition_samples += 1

                        # -----------------------------------------
                        # NEW PERSON
                        # -----------------------------------------

                        else:

                            confirmed_name = (
                                recognized_name
                            )

                            recognition_samples = 1

                            # Important:
                            # Reset previous access state.
                            access_granted_sent = False

                        # -----------------------------------------
                        # RECOGNITION CONFIRMED
                        # -----------------------------------------

                        if (
                            recognition_samples
                            >= RECOGNITION_CONFIRMATION_SAMPLES
                        ):

                            log.info(
                                "Face recognized: %s",
                                recognized_name,
                            )

                            challenge = (
                                liveness.start_new_challenge()
                            )

                            event_pub.publish_json(
                                {
                                    "event": "recognition_confirmed",
                                    "name": recognized_name,
                                    "challenge": challenge,
                                }
                            )

                    else:

                        confirmed_name = None

                        recognition_samples = 0

                # =================================================
                # LIVENESS CHALLENGE
                # =================================================

                elif primary_face is not None:

                    pose = measure_head_pose(
                        primary_face
                    )

                    if pose is not None:

                        liveness.evaluate(
                            pose
                        )

                    # =============================================
                    # LIVENESS SUCCESS
                    # =============================================

                    if (
                        liveness.is_passed
                        and not access_granted_sent
                    ):

                        access_granted_sent = True

                        log.info(
                            "ACCESS GRANTED: %s",
                            confirmed_name,
                        )

                        event_pub.publish_json(
                            {
                                "event": "access_granted",
                                "name": confirmed_name,
                            }
                        )

                        # -----------------------------------------
                        # Start cooldown.
                        # -----------------------------------------

                        cooldown_until = (
                            time.monotonic()
                            + ACCESS_COOLDOWN_SECONDS
                        )

                    # =============================================
                    # LIVENESS FAILURE
                    # =============================================

                    elif liveness.is_failed:

                        log.warning(
                            "ACCESS DENIED: liveness failed"
                        )

                        event_pub.publish_json(
                            {
                                "event": "liveness_failed",
                                "name": confirmed_name,
                            }
                        )

                        # -----------------------------------------
                        # Start cooldown.
                        # -----------------------------------------

                        cooldown_until = (
                            time.monotonic()
                            + ACCESS_COOLDOWN_SECONDS
                        )

                        # -----------------------------------------
                        # Reset authentication.
                        # -----------------------------------------

                        liveness.reset(
                            log_reset=False
                        )

                        confirmed_name = None

                        recognition_samples = 0

                        access_granted_sent = False

            # =================================================
            # NO FACE
            # =================================================

            elif face_active:

                no_face_checks += 1

                if (
                    no_face_checks
                    >= CLEAR_ACTIVE_AFTER_NO_FACE_CHECKS
                ):

                    face_active = False

                    no_face_checks = 0

                    confirmed_name = None

                    recognition_samples = 0

                    access_granted_sent = False

                    liveness.reset()

                    log.info(
                        "Face state cleared"
                    )

                    event_pub.publish_json(
                        {
                            "event": "face_lost"
                        }
                    )

        # =====================================================
        # SHUTDOWN
        # =====================================================

        except KeyboardInterrupt:

            log.info(
                "Vision Worker shutting down..."
            )

            break

        except Exception as e:

            log.exception(
                "Error in vision loop: %s",
                e,
            )

    # =========================================================
    # CLOSE ZMQ
    # =========================================================

    frame_sub.close()
    anno_pub.close()
    event_pub.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_vision_worker()
# FaceRecognition: Project and Architecture Guide

## What this project is

FaceRecognition is a local, real-time access-verification prototype. A Raspberry Pi captures a webcam feed and sends JPEG frames to a laptop. The laptop detects a face, rejects masks and likely spoof attempts, compares a clear face with a small local gallery, then requires an active head-movement liveness challenge before publishing an `access_granted` event.

It is designed for low latency on a local network. It is not currently a training pipeline, cloud service, or complete production access-control system.

## Current architecture

```text
Raspberry Pi / USB webcam
  stream.py (JPEG/MJPEG over WebSocket)
          |
          v
Laptop gateway: laptopserver.py (FastAPI)
  /ws/video receives frames; /ws/view sends viewer frames
          |
          v
ZeroMQ IPC bus in /tmp
  camera.frames (raw JPEGs)
          |
          v
Vision worker: src/workers/vision_worker.py
  YuNet -> mask -> anti-spoof -> SFace -> active liveness
          |
          +--> camera.annotated --> gateway --> browser viewer
          +--> vision.events (security and access events)
```

The three runtime processes are intentionally separate:

1. `stream.py` runs on the Pi and reconnects when the laptop is unavailable.
2. `laptopserver.py` runs the HTTP/WebSocket gateway on the laptop.
3. `src/workers/vision_worker.py` runs the computer-vision pipeline on the laptop.

The gateway publishes raw frames on `camera.frames`. The worker subscribes with ZeroMQ `CONFLATE`, so it works on the newest available frame rather than accumulating latency. It publishes annotated JPEGs on `camera.annotated`; each browser connection consumes that stream with the same newest-frame behavior. `vision.events` is available for a future access-control, logging, or alerting consumer.

### Decision flow

On sampled frames (at most every 0.2 seconds), the worker:

1. Decodes the JPEG and selects the first YuNet face detection.
2. Rejects a detected mask using `mask_detector.onnx`.
3. Rejects a likely presentation attack with MiniFASNetV2 (`real_score > 0.6`).
4. Aligns the face with YuNet landmarks and creates an SFace embedding.
5. Compares against one enrolled image per identity in `known_faces/`; cosine similarity must be at least `0.36`.
6. Requires three consecutive sampled recognitions of the same known identity.
7. Chooses a random prompt—look left, right, up, or down. After a neutral-pose baseline, two matching movement samples within seven seconds pass liveness.
8. Emits `access_granted` when liveness succeeds.

Frames received between AI samples are forwarded directly, keeping the viewer responsive. Known-face crops are saved at most once per identity every ten seconds; mask/spoof frames at most every five seconds.

## Key components

| Path | Responsibility |
| --- | --- |
| `stream.py` | Pi camera capture, MJPEG passthrough when available, WebSocket transmission, reconnect loop. |
| `laptopserver.py` | FastAPI ingestion/viewer endpoints and bus gateway. |
| `src/core/bus.py` | Small ZeroMQ PUB/SUB wrapper using local IPC sockets. |
| `src/workers/vision_worker.py` | Sampling, recognition state, liveness, annotations, captures, and event publication. |
| `src/detect/` | YuNet detection, SFace recognition, anti-spoofing, and mask detection wrappers. |
| `models/` | ONNX models required at runtime. |
| `known_faces/` | Local reference photos; each filename stem is an identity. |
| `captures/`, `suspicious/` | Runtime evidence output for recognized and suspicious activity. |

## What changed from the old design

The current uncommitted architecture refactor replaces an earlier single-process FastAPI design. Previously, `laptopserver.py` owned the WebSocket gateway, in-memory `asyncio.Queue` fan-out, model loading, and AI worker task. It now acts only as the gateway, while vision logic lives in `src/workers/vision_worker.py` and the two communicate over ZeroMQ IPC.

This improves separation of concerns: slow inference or an AI failure is less coupled to WebSocket handling, and future local consumers can subscribe to raw frames or events without editing the gateway. The refactor adds `pyzmq` as a dependency.

## Strengths

- **Low-latency design:** Pi-side MJPEG passthrough, TCP_NODELAY, small camera buffering, frame conflation, and a 5 FPS AI sampling cap prioritize freshness.
- **Clear boundaries:** capture, transport, gateway, and vision inference are separate processes with small interfaces.
- **Extensible event channel:** `vision.events` can support a door controller, audit sink, notifications, or dashboard.
- **Layered verification:** mask detection, anti-spoofing, recognition confirmation, and active liveness are stronger than recognition alone.
- **Local-first privacy:** processing and enrolled images remain on the local network by default.
- **Simple enrollment:** a clear image in `known_faces/` is enough to enroll an identity.
- **Some resilience:** the Pi reconnects automatically; slow viewers/workers do not build stale-frame queues.

## Limitations and risks

- **Single point of failure:** the laptop, gateway, worker, and IPC sockets are essential; there is no supervision, health check, restart policy, or failover.
- **No authentication or encryption:** endpoints are unauthenticated and the Pi configuration uses `ws://`; reachable-network users could view or inject video.
- **No durable audit trail:** events are transient; there is no database, retained decision log, or model/config version attached to a decision.
- **No actuator consumer:** `access_granted` is published, but nothing yet unlocks a door or records an outcome.
- **First-face-only processing:** multiple faces are not rejected or managed deterministically, which is unsuitable for a security boundary.
- **Uncalibrated static thresholds:** similarity, spoof, mask, and pose thresholds are hard-coded; false-accept and false-reject performance is unknown for the target environment.
- **Lightweight liveness:** head movement helps against a still image but does not by itself guarantee resistance to video replay, deepfakes, or capable presentation attacks.
- **IPC lifecycle and delivery:** fixed `/tmp` socket paths can survive an unclean shutdown; PUB/SUB drops messages for disconnected/not-yet-subscribed consumers. Good for live video, unsuitable for must-deliver security events.
- **Configuration gaps:** the laptop IP, camera settings, paths, and thresholds require source edits; there is no config-file or environment layer.
- **Biometric-data handling:** enrolled, captured, and suspicious images need consent, access, retention, encryption, and deletion policies before real deployment.

## Documentation and test status

The older Markdown files were reviewed:

- `architecture.md` correctly describes the Pi-to-laptop streaming goal, but its in-process queue broadcaster is now outdated by ZeroMQ.
- The prior `project.md` described a generic training/evaluation repository (`src/train`, `configs`, `experiments`, and `requirements.txt`). Those workflows do not exist here, so this document replaces it with an implementation-based guide.
- `agent.md` is a generic proposal for data-prep, training, evaluation, CI, and monitoring agents. Its referenced agents and paths do not exist yet; treat it as future ideas, not operational documentation.
- `README.md` is empty.

`tests/testbus.py` is a manual multiprocessing smoke script. `tests/test_liveness.py` still imports liveness classes from `laptopserver.py`, while the refactor moved them to `src/workers/vision_worker.py`; it needs updating before it can validate the new layout. This document records that mismatch only and makes no code or test change.

## Running the prototype

Install dependencies, then run these as separate processes on the laptop:

```bash
uv run python laptopserver.py
uv run python -m src.workers.vision_worker
```

Open `http://<laptop-ip>:8000/` in a browser. On the Pi, set `LAPTOP_WS_URL` in `stream.py` to the laptop's LAN IP, then run:

```bash
uv run python stream.py
```

The deployment needs the ONNX files in `models/`, a usable enrollment image in `known_faces/`, a reachable local network, and compatible OpenCV camera support. Project metadata declares Python 3.14 or later; verify that each target has a compatible runtime.

## Recommended next steps

1. Move URLs, thresholds, output paths, and camera settings to configuration.
2. Add authentication, TLS/WSS, and network restrictions before any non-lab deployment.
3. Add a durable event consumer and a fail-closed policy for worker, model, and transport failures.
4. Explicitly reject/manage multiple faces and calibrate all decision thresholds with representative data.
5. Add process supervision, health endpoints, structured logs, metrics for FPS/drops/latency, and IPC cleanup.
6. Repair and expand tests around the new module boundary and end-to-end flow.
7. Establish consent, retention, encryption, and deletion practices for biometric data.

## Scope note

This guide reflects the files currently in the working tree, including the in-progress ZeroMQ refactor. It deliberately does not claim that the generic training/agent workflows in the legacy documentation are implemented.

# Real-Time Face Verification

A local face-verification prototype that streams camera and microphone data
from a Raspberry Pi to a laptop. The laptop runs face detection, mask
detection, anti-spoofing, face recognition, and active head-movement liveness
verification.

This project is a prototype and is not production-ready access-control
software. Hardware validation, authentication, encryption, threshold
calibration, and biometric-data retention controls are still required.

## Architecture

```text
Raspberry Pi camera and microphone
	|
	| WebSocket video/audio streams
	v
FastAPI gateway (laptopserver.py)
	|
	| ZeroMQ IPC sockets in /tmp
	v
Vision worker (src/workers/vision_worker.py)
	|
	+-- YuNet face detection and landmarks
	+-- Face-mask classification
	+-- MiniFASNetV2 anti-spoofing
	+-- SFace face recognition
	+-- Active head-movement liveness challenge
```

## Features

- Real-time JPEG video streaming from a Raspberry Pi.
- FastAPI WebSocket gateway for video, audio, and browser viewing.
- ZeroMQ publisher/subscriber communication between gateway and workers.
- YuNet face detection with five facial landmarks.
- SFace embedding comparison using cosine similarity.
- Face-mask classification.
- MiniFASNetV2 presentation-attack detection.
- Active liveness verification using a random head-movement challenge.
- Annotated live feed for browser viewing.

## Models

The application expects these files under `models/`:

| File | Purpose |
| --- | --- |
| `face_detection_yunet.onnx` | Face detection and facial landmarks |
| `face_recognition_sface.onnx` | Face embeddings and identity matching |
| `2.7_80x80_MiniFASNetV2.onnx` | Real-face and spoof classification |
| `mask_detector.onnx` | Masked and unmasked face classification |

Model files are ignored for future commits to keep the source repository
clean. They are required at runtime and must be obtained separately from their
original sources. Verify the license and checksum of every model before use.

The current repository history may contain model files from an earlier
snapshot. The ignore rule only affects future additions; it does not remove
files already tracked by Git.

## Requirements

- Python 3.14 or newer, as declared in `pyproject.toml`.
- A Raspberry Pi with a compatible camera and microphone for the Pi client.
- A laptop on the same local network.
- OpenCV with ONNX support.
- The required model files in `models/`.
- A reference face image in `known_faces/` for local enrollment.

## Installation

Install the Python dependencies with `uv`:

```bash
uv sync
```

The Raspberry Pi client also requires a working PortAudio/PyAudio system
installation for microphone capture.

## Running

Start the laptop gateway:

```bash
uv run python laptopserver.py
```

In a second laptop terminal, start the vision worker:

```bash
uv run python -m src.workers.vision_worker
```

Open the live viewer at:

```text
http://<laptop-ip>:8000/
```

Set the laptop LAN address in `stream.py`, then run the Pi client:

```bash
uv run python stream.py
```

The gateway and worker must run as separate processes. The Pi and laptop must
be reachable on the same local network.

## Data and privacy

Do not commit enrolled face images or runtime captures. The following folders
are ignored by Git:

- `known_faces/`
- `captures/`
- `suspicious/`

These files contain biometric or potentially sensitive data. Use consent,
limited retention, access controls, and secure deletion when handling real
images.

## Limitations

- No authentication or TLS is configured for the WebSocket endpoints.
- Laptop IP address and several runtime settings are hard-coded.
- Multiple faces are not handled as a security policy.
- Decision thresholds are not calibrated for a target environment.
- There is no durable audit log or door-actuator integration.
- Final hardware and end-to-end validation is pending.

## Project status

Prototype implementation archived for portfolio use. Further hardware testing
and production hardening are planned when the required hardware is available.

"""
Runs on the Raspberry Pi 3B.
Captures frames from the USB webcam, JPEG-compresses them, and streams
them to the laptop over a WebSocket. Reconnects automatically if the
laptop server restarts or the connection drops.
"""
import asyncio
import socket
import time
import cv2
import websockets
from websockets.exceptions import WebSocketException

LAPTOP_WS_URL = "ws://192.168.1.50:8000/ws/video"  # replace with the laptop's local IP
CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480  # this camera's max supported resolution
TARGET_FPS = 10
JPEG_QUALITY = 80  # only used when the camera can't hand back MJPEG directly
ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
FRAME_INTERVAL = 1.0 / TARGET_FPS


def open_camera():
    """Returns (cap, passthrough). When passthrough is True, cap.read() hands back
    already-JPEG-encoded bytes straight from the camera's own MJPEG mode, so we
    skip OpenCV's decode-then-re-encode round trip — the biggest CPU/latency cost
    on a Pi 3B."""
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)  # explicit backend so BUFFERSIZE is honored
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always grab the newest frame, not a stale one
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)  # ask for raw MJPEG bytes instead of decoded BGR
    if not cap.isOpened():
        raise RuntimeError("Could not open the USB webcam")
    passthrough = int(cap.get(cv2.CAP_PROP_FOURCC)) == cv2.VideoWriter_fourcc(*"MJPG")
    if not passthrough:
        print("Camera doesn't support MJPEG passthrough — falling back to re-encode")
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    return cap, passthrough


async def grab_and_encode(cap, passthrough: bool):
    """Runs the blocking OpenCV calls off the event loop."""
    ok, frame = await asyncio.to_thread(cap.read)
    if not ok:
        return None
    if passthrough:
        return frame.tobytes()  # already JPEG — no re-encode needed
    ok, buffer = await asyncio.to_thread(cv2.imencode, ".jpg", frame, ENCODE_PARAMS)
    if not ok:
        return None
    return buffer.tobytes()


async def stream_video(cap, passthrough: bool):
    """Owns one websocket connection; camera lifetime is managed by the caller."""
    async with websockets.connect(
        LAPTOP_WS_URL,
        max_size=2 ** 22,
        ping_interval=5,
        ping_timeout=10,
    ) as ws:
        # Disable Nagle's algorithm — these are small, frequent sends, and
        # letting TCP buffer/coalesce them adds visible lag per frame.
        sock = ws.transport.get_extra_info("socket")
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

        print("Connected to laptop — streaming started")
        while True:
            loop_start = time.monotonic()

            jpeg_bytes = await grab_and_encode(cap, passthrough)
            if jpeg_bytes is None:
                await asyncio.sleep(0.1)
                continue

            await ws.send(jpeg_bytes)

            elapsed = time.monotonic() - loop_start
            await asyncio.sleep(max(0.0, FRAME_INTERVAL - elapsed))


async def run_forever():
    """Keeps retrying the connection so a laptop restart doesn't kill the Pi client.
    The camera is opened once and reused across reconnects."""
    cap, passthrough = open_camera()
    try:
        while True:
            try:
                await stream_video(cap, passthrough)
            except (OSError, WebSocketException) as exc:
                print(f"Connection lost ({exc}) — retrying in 3s")
                await asyncio.sleep(3)
            except cv2.error as exc:
                print(f"Camera error ({exc}) — reopening camera in 3s")
                cap.release()
                await asyncio.sleep(3)
                cap, passthrough = open_camera()
    finally:
        cap.release()


if __name__ == "__main__":
    try:
        asyncio.run(run_forever())
    except KeyboardInterrupt:
        print("\nStopped.")

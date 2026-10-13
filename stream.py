"""
Runs on the Raspberry Pi 3B.
Captures video frames and webcam microphone audio concurrently, streaming
them to the laptop over separate WebSocket channels. Automatically reconnects
if the connection drops.
"""
import asyncio
import socket
import time
import cv2
import pyaudio
import websockets
from websockets.exceptions import WebSocketException

# Laptop Configuration
LAPTOP_IP = "192.168.1.50"  # Replace with your laptop's local IP
VIDEO_WS_URL = f"ws://{LAPTOP_IP}:8000/ws/video"
AUDIO_WS_URL = f"ws://{LAPTOP_IP}:8000/ws/audio_in"

# Video Configuration
CAPTURE_WIDTH, CAPTURE_HEIGHT = 640, 480
TARGET_FPS = 10
JPEG_QUALITY = 80
ENCODE_PARAMS = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
FRAME_INTERVAL = 1.0 / TARGET_FPS

# Audio Configuration (16kHz Mono PCM for AI models)
AUDIO_FORMAT = pyaudio.paInt16
AUDIO_CHANNELS = 1
AUDIO_RATE = 16000
AUDIO_CHUNK = 1024


# --- VIDEO FUNCTIONS ---

def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    
    if not cap.isOpened():
        raise RuntimeError("Could not open the USB webcam")
        
    passthrough = int(cap.get(cv2.CAP_PROP_FOURCC)) == cv2.VideoWriter_fourcc(*"MJPG")
    if not passthrough:
        print("Camera doesn't support MJPEG passthrough — falling back to re-encode")
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
    return cap, passthrough


async def grab_and_encode(cap, passthrough: bool):
    ok, frame = await asyncio.to_thread(cap.read)
    if not ok:
        return None
    if passthrough:
        return frame.tobytes()
    ok, buffer = await asyncio.to_thread(cv2.imencode, ".jpg", frame, ENCODE_PARAMS)
    if not ok:
        return None
    return buffer.tobytes()


async def stream_video_loop():
    """Manages camera opening, streaming, and reconnection loops."""
    while True:
        try:
            cap, passthrough = open_camera()
            print("[Video] Camera opened. Connecting to server...")
            
            async with websockets.connect(
                VIDEO_WS_URL, max_size=2**22, ping_interval=5, ping_timeout=10
            ) as ws:
                sock = ws.transport.get_extra_info("socket")
                if sock is not None:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)

                print("[Video] Connected — streaming video started")
                while True:
                    loop_start = time.monotonic()
                    jpeg_bytes = await grab_and_encode(cap, passthrough)
                    if jpeg_bytes is None:
                        await asyncio.sleep(0.1)
                        continue

                    await ws.send(jpeg_bytes)
                    elapsed = time.monotonic() - loop_start
                    await asyncio.sleep(max(0.0, FRAME_INTERVAL - elapsed))

        except (OSError, WebSocketException) as exc:
            print(f"[Video] Connection lost ({exc}) — retrying in 3s")
            await asyncio.sleep(3)
        except cv2.error as exc:
            print(f"[Video] Camera error ({exc}) — reopening in 3s")
            await asyncio.sleep(3)
        finally:
            if 'cap' in locals() and cap.isOpened():
                cap.release()


# --- AUDIO FUNCTIONS ---

def find_input_device_index(p: pyaudio.PyAudio) -> int:
    """Finds the first input-capable device (USB Webcam Mic)."""
    for i in range(p.get_device_count()):
        dev_info = p.get_device_info_by_index(i)
        if dev_info.get("maxInputChannels", 0) > 0:
            print(f"[Audio] Found input device [{i}]: {dev_info.get('name')}")
            return i
    raise RuntimeError("No input microphone found on Raspberry Pi!")

async def stream_audio_loop():
    p = pyaudio.PyAudio()
    
    while True:
        try:
            device_index = find_input_device_index(p)
            print(f"[Audio] Opening mic stream on device index {device_index}...")
            
            stream = p.open(
                format=AUDIO_FORMAT,
                channels=AUDIO_CHANNELS,
                rate=AUDIO_RATE,
                input=True,
                input_device_index=device_index, # Explicitly use the webcam mic
                frames_per_buffer=AUDIO_CHUNK
            )
            
            print("[Audio] Connecting to server...")
            async with websockets.connect(
                AUDIO_WS_URL, ping_interval=5, ping_timeout=10
            ) as ws:
                print("[Audio] Connected — streaming microphone started")
                while True:
                    pcm_data = await asyncio.to_thread(
                        stream.read, AUDIO_CHUNK, False
                    )
                    await ws.send(pcm_data)

        except (OSError, WebSocketException, RuntimeError) as exc:
            print(f"[Audio] Error ({exc}) — retrying in 3s")
            await asyncio.sleep(3)
        finally:
            if 'stream' in locals():
                stream.stop_stream()
                stream.close()
    p.terminate()

# --- MAIN RUNNER ---

async def main():
    # Run both video and audio loops concurrently
    await asyncio.gather(
        stream_video_loop(),
        stream_audio_loop()
    )

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped Pi Client.")
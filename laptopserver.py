import asyncio
import logging
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, responses

from src.core.bus import ZmqPublisher, ZmqSubscriber

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gateway")

app = FastAPI()

# Global publisher for the incoming camera frames
raw_pub = ZmqPublisher("camera.frames")

# Add this near the top with your other publishers
audio_pub = ZmqPublisher("audio.raw")

@app.websocket("/ws/audio_in")
async def ingest_audio(websocket: WebSocket):
    await websocket.accept()
    log.info("Pi connected — ingesting microphone audio")
    try:
        while True:
            # Receive raw PCM audio chunks from the Pi
            audio_chunk = await websocket.receive_bytes()
            # Push to the ZeroMQ bus for the AI/Speaker workers
            await asyncio.to_thread(audio_pub.publish_bytes, audio_chunk)
    except WebSocketDisconnect:
        log.info("Pi audio disconnected")

    
@app.websocket("/ws/video")
async def ingest_video(websocket: WebSocket):
    await websocket.accept()
    log.info("Pi connected — ingesting video")
    frame_count = 0
    window_start = time.monotonic()
    try: 
        while True:
            frame = await websocket.receive_bytes()
            # Push raw frame to the ZeroMQ bus
            await asyncio.to_thread(raw_pub.publish_bytes, frame)
            
            frame_count += 1
            elapsed = time.monotonic() - window_start
            if elapsed >= 5.0:
                log.info("Camera ingest FPS: %.1f", frame_count / elapsed)
                frame_count = 0
                window_start = time.monotonic()
    except WebSocketDisconnect:
        log.info("Pi disconnected")

@app.websocket("/ws/view")
async def view_video(websocket: WebSocket):
    await websocket.accept()
    log.info("Browser connected — streaming annotated feed")
    # Local subscriber that drops old frames to stay in real-time
    anno_sub = ZmqSubscriber("camera.annotated", conflate=True)
    try:
        while True:
            # Pull annotated frames from the ZeroMQ bus
            frame = await asyncio.to_thread(anno_sub.receive_bytes)
            await websocket.send_bytes(frame)
    except WebSocketDisconnect:
        log.info("Browser disconnected")
    finally:
        anno_sub.close()

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
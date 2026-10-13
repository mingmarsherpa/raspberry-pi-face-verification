import pyaudio
import logging
from src.core.bus import ZmqSubscriber

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("speaker_worker")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

def run_speaker():
    log.info("Initializing Laptop Speaker Worker...")
    
    # Subscribe to the raw audio coming from the mic to test the pipeline
    # In Phase 4, we will change this topic to "audio.tts"
    audio_sub = ZmqSubscriber("audio.raw", conflate=False)
    
    p = pyaudio.PyAudio()
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        output=True,
        frames_per_buffer=CHUNK
    )
    
    log.info("Speaker ready. Listening to bus...")
    
    try:
        while True:
            # Block until we receive an audio chunk from ZeroMQ
            audio_chunk = audio_sub.receive_bytes()
            # Play it through the laptop speakers
            stream.write(audio_chunk)
    except KeyboardInterrupt:
        log.info("Speaker Worker stopping...")
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()
        audio_sub.close()

if __name__ == "__main__":
    run_speaker()
import time
import multiprocessing
from src.core.bus import ZmqPublisher, ZmqSubscriber

def run_publisher():
    pub = ZmqPublisher("test_topic")
    time.sleep(0.5) # Give subscriber time to connect
    print("[PUB] Sending event...")
    pub.publish_json({"event": "system_start", "status": "clear"})
    time.sleep(0.1)
    pub.close()

def run_subscriber():
    sub = ZmqSubscriber("test_topic")
    print("[SUB] Waiting for event...")
    data = sub.receive_json()
    print(f"[SUB] Received: {data}")
    sub.close()

if __name__ == "__main__":
    p1 = multiprocessing.Process(target=run_subscriber)
    p2 = multiprocessing.Process(target=run_publisher)
    
    p1.start()
    p2.start()
    
    p1.join()
    p2.join()
    print("ZeroMQ Bus Test Complete.")
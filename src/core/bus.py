import json
import zmq
import logging

log = logging.getLogger("zmq_bus")

class ZmqPublisher:
    def __init__(self, topic: str):
        self.topic = topic
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.PUB)
        # Bind creates the IPC socket file in /tmp/
        self.ipc_path = f"ipc:///tmp/{self.topic}.ipc"
        self.socket.bind(self.ipc_path)
        log.info(f"Publisher bound to {self.ipc_path}")

    def publish_bytes(self, data: bytes):
        """Used for high-fps video frames or raw audio chunks."""
        self.socket.send(data)

    def publish_json(self, data: dict):
        """Used for sending structured events and states."""
        self.socket.send_json(data)

    def close(self):
        self.socket.close()


class ZmqSubscriber:
    def __init__(self, topic: str, conflate: bool = False):
        self.topic = topic
        self.context = zmq.Context.instance()
        self.socket = self.context.socket(zmq.SUB)
        
        # Conflate drops old messages if the subscriber falls behind.
        # Set to True for video frames (always want the newest), False for events.
        if conflate:
            self.socket.setsockopt(zmq.CONFLATE, 1)
            
        self.ipc_path = f"ipc:///tmp/{self.topic}.ipc"
        self.socket.connect(self.ipc_path)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        log.info(f"Subscriber connected to {self.ipc_path}")

    def receive_bytes(self, flags=0) -> bytes:
        return self.socket.recv(flags=flags)

    def receive_json(self, flags=0) -> dict:
        return self.socket.recv_json(flags=flags)

    def close(self):
        self.socket.close()
# Architecture: Pi → Laptop → Viewers (WebSocket)

Brief

A simple local streaming design: Raspberry Pi captures frames, streams JPEGs via WebSocket to the laptop server (FastAPI). The laptop keeps latest frame and fans out to viewers and optional AI workers via per-subscriber queues (maxsize=1).

Pros

- Low latency: local WS and MJPEG passthrough keep end-to-end delay small.
- Backpressure-safe: queues of size 1 drop stale frames, preventing slow consumers from adding lag.
- CPU-efficient on Pi: passthrough avoids decode/re-encode when camera supplies MJPEG.
- Modular: AI worker can subscribe without changing broadcaster logic.
- Simple to run and debug: few moving parts, easy local testing.
- Privacy-friendly: no cloud by default; data stays on local network.

Cons

- Single point of failure: laptop server is central; its crash stops the system.
- Scalability limits: laptop may not handle many heavy AI workers or many viewers.
- No persistence: frames are transient; no easy replay or audit trail.
- Security gaps: no built-in auth, encryption depends on deployment (ws vs wss).
- Network fragility: relies on local IPs; NAT/firewall complicate remote access.
- Limited observability: no built-in metrics/alerts — need extra tooling for production.

Quick mitigations (short)

- HA: add redundant ingest/edge brokers (RTMP, MQTT, or a lightweight broker) or move to a cloud relay.
- Scale: offload AI to separate worker hosts or use message broker with horizontal consumers.
- Persistence: add optional recorder to write frames or sampled embeddings to disk/DB.
- Security: enable TLS (wss), add auth tokens, and limit network access.
- Observability: emit simple metrics (fps, queue drops, errors) to Prometheus or logs.

Done. See [architecture.md](/home/mingmarsherpa/facerecognition/architecture.md).
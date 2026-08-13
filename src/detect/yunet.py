import cv2
import numpy as np
from typing import List, Tuple, Any


class YunetDetector:
    """
    Wrapper for OpenCV YuNet Face Detector.
    Provides raw detections (bounding box + landmarks) required by SFace for alignment.
    """
    def __init__(self, model_path: str = "models/face_detection_yunet.onnx", conf_threshold: float = 0.6):
        self.conf_threshold = conf_threshold
        self.input_size = (320, 320)
        
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV installed without FaceDetectorYN. Upgrade opencv-python.")

        self.net = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=self.input_size,
            score_threshold=self.conf_threshold,
            nms_threshold=0.3,
            top_k=5000
        )

    def detect_raw(self, bgr_frame: np.ndarray) -> Any:
        """Returns raw detection array containing bounding boxes and 5 facial landmarks."""
        h, w, _ = bgr_frame.shape
        if (w, h) != self.input_size:
            self.input_size = (w, h)
            self.net.setInputSize((w, h))
        
        _, faces = self.net.detect(bgr_frame)
        return faces if faces is not None else []

    def detect(self, bgr_frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Returns simplified list of bounding boxes [(x, y, w, h), ...]"""
        raw_faces = self.detect_raw(bgr_frame)
        if len(raw_faces) == 0:
            return []
        return [tuple(map(int, face[:4])) for face in raw_faces]
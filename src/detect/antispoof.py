import os
import cv2
import numpy as np

class AntiSpoofDetector:
    def __init__(self, model_path: str = "models/2.7_80x80_MiniFASNetV2.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Anti-spoof model missing at {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def is_real(self, frame: np.ndarray, bbox: tuple) -> tuple[bool, float]:
        h_img, w_img, _ = frame.shape
        x, y, w, h = map(int, bbox)

        # MiniFASNet needs a 2.7x scaled bounding box for context
        cx, cy = x + w // 2, y + h // 2
        scale = int(max(w, h) * 1.35)

        x1, y1 = max(0, cx - scale), max(0, cy - scale)
        x2, y2 = min(w_img, cx + scale), min(h_img, cy + scale)

        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            return False, 0.0

        blob = cv2.dnn.blobFromImage(face_crop, 1.0, (80, 80), (0, 0, 0), swapRB=False)
        self.net.setInput(blob)
        preds = self.net.forward()[0]
        
        # Softmax computation
        exp_preds = np.exp(preds - np.max(preds))
        probs = exp_preds / np.sum(exp_preds)
        
        real_score = probs[1] # Index 1 is Real, Index 0 is Spoof
        return real_score > 0.6, float(real_score)
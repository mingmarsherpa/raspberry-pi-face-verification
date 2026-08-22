import os
import cv2
import numpy as np

class MaskDetector:
    def __init__(self, model_path: str = "models/mask_detector.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Mask detector model missing at {model_path}")
        self.net = cv2.dnn.readNetFromONNX(model_path)

    def is_masked(self, frame: np.ndarray, bbox: tuple) -> bool:
        h_img, w_img, _ = frame.shape
        x, y, w, h = map(int, bbox)
        
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(w_img, x + w), min(h_img, y + h)
        face_crop = frame[y1:y2, x1:x2]
        
        if face_crop.size == 0 or face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            return False

        blob = cv2.dnn.blobFromImage(face_crop, 1.0/255.0, (224, 224), (0, 0, 0), swapRB=True)
        self.net.setInput(blob)
        preds = self.net.forward()[0]
        
        # Index 0 is Masked, Index 1 is Unmasked
        return preds[0] > preds[1]
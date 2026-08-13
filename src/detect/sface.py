import os
from typing import Optional

import cv2
import numpy as np

from src.detect.yunet import YunetDetector


class SFaceRecognizer:
    def __init__(
        self,
        model_path: str = "models/face_recognition_sface.onnx",
        known_faces_dir: str = "known_faces",
        threshold: float = 0.36,
        detector: Optional[YunetDetector] = None,
    ):
        if not os.path.exists(model_path):
            legacy_path = "models/face_recognition_sface_2021dec.onnx"
            if os.path.exists(legacy_path):
                model_path = legacy_path

        self.recognizer = cv2.FaceRecognizerSF.create(model_path, "")
        self.known_faces_dir = known_faces_dir
        self.threshold = threshold
        self.detector = detector or YunetDetector()
        self.known_features = {}
        self._load_known_faces()

    def _load_known_faces(self):
        if not os.path.exists(self.known_faces_dir):
            return

        for file in os.listdir(self.known_faces_dir):
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                name = os.path.splitext(file)[0]
                path = os.path.join(self.known_faces_dir, file)
                img = cv2.imread(path)
                if img is None:
                    continue

                # Detect facial landmarks and align image before generating feature embeddings
                raw_faces = self.detector.detect_raw(img)
                if len(raw_faces) == 0:
                    print(f"[SFace] Warning: No face detected in '{file}'. Skipping.")
                    continue

                primary_face = raw_faces[0]
                aligned_face = self.recognizer.alignCrop(img, primary_face)
                feature = self.recognizer.feature(aligned_face)
                self.known_features[name] = feature
                print(f"[SFace] Enrolled known face: '{name}'")

    def recognize(self, img: np.ndarray, primary_face: np.ndarray):
        aligned_face = self.recognizer.alignCrop(img, primary_face)
        current_feature = self.recognizer.feature(aligned_face)

        best_name = "unknown"
        best_score = 0.0

        for name, known_feature in self.known_features.items():
            # cv2.FaceRecognizerSF_FR_COSINE evaluates cosine similarity (0-1 range)
            score = self.recognizer.match(
                known_feature, current_feature, cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > best_score:
                best_score = score
                if score >= self.threshold:
                    best_name = name

        return best_name, best_score
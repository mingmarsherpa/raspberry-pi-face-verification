import unittest

import numpy as np

from laptopserver import ActiveLivenessManager, HeadPoseMeasurement, measure_head_pose


class HeadPoseMeasurementTests(unittest.TestCase):
    def test_measurement_is_normalized_from_yunet_landmarks(self):
        # bounding box, right eye, left eye, nose, right mouth, left mouth, score
        face = np.array([0, 0, 100, 100, 40, 40, 60, 40, 50, 52, 42, 70, 58, 70, 0.9])

        pose = measure_head_pose(face)

        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.yaw, 0.0)
        self.assertAlmostEqual(pose.pitch, 0.4)

    def test_degenerate_landmarks_are_rejected(self):
        face = np.zeros(15)
        self.assertIsNone(measure_head_pose(face))


class ActiveLivenessManagerTests(unittest.TestCase):
    neutral = HeadPoseMeasurement(yaw=0.0, pitch=0.4)

    def _ready_manager(self, challenge: str) -> ActiveLivenessManager:
        manager = ActiveLivenessManager(
            timeout_seconds=1.0,
            baseline_samples=1,
            required_movement_samples=2,
        )
        manager.start_new_challenge()
        manager.current_challenge = challenge
        self.assertFalse(manager.evaluate(self.neutral, now=10.0))
        return manager

    def test_each_direction_requires_two_matching_samples(self):
        movements = {
            "look left": HeadPoseMeasurement(yaw=0.2, pitch=0.4),
            "look right": HeadPoseMeasurement(yaw=-0.2, pitch=0.4),
            "look up": HeadPoseMeasurement(yaw=0.0, pitch=0.2),
            "look down": HeadPoseMeasurement(yaw=0.0, pitch=0.6),
        }

        for challenge, movement in movements.items():
            with self.subTest(challenge=challenge):
                manager = self._ready_manager(challenge)
                self.assertFalse(manager.evaluate(movement, now=10.1))
                self.assertTrue(manager.evaluate(movement, now=10.2))
                self.assertTrue(manager.is_passed)

    def test_opposite_or_one_frame_movement_does_not_pass(self):
        manager = self._ready_manager("look left")
        self.assertFalse(manager.evaluate(HeadPoseMeasurement(yaw=-0.2, pitch=0.4), now=10.1))
        self.assertFalse(manager.evaluate(HeadPoseMeasurement(yaw=0.2, pitch=0.4), now=10.2))
        self.assertFalse(manager.is_passed)

    def test_timeout_marks_attempt_failed(self):
        manager = self._ready_manager("look left")
        self.assertFalse(manager.evaluate(self.neutral, now=11.1))
        self.assertTrue(manager.is_failed)
        self.assertFalse(manager.is_active)

    def test_reset_clears_completed_session(self):
        manager = self._ready_manager("look left")
        movement = HeadPoseMeasurement(yaw=0.2, pitch=0.4)
        manager.evaluate(movement, now=10.1)
        manager.evaluate(movement, now=10.2)

        manager.reset(log_reset=False)

        self.assertIsNone(manager.current_challenge)
        self.assertFalse(manager.is_passed)


if __name__ == "__main__":
    unittest.main()

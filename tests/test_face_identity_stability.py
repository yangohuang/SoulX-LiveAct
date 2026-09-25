import unittest
from types import SimpleNamespace

import numpy as np

from face_identity_stability import (
    cosine, detect_single_face_embedding, summarize_embeddings,
)


class FaceIdentityStabilityTests(unittest.TestCase):
    def test_cosine_reports_orthogonal_and_matching_embeddings(self):
        self.assertAlmostEqual(cosine(np.array([1., 0.]), np.array([2., 0.])), 1.)
        self.assertAlmostEqual(cosine(np.array([1., 0.]), np.array([0., 1.])), 0.)

    def test_missing_detection_is_counted_not_imputed(self):
        reference = np.array([1., 0.])
        scores = summarize_embeddings(
            [np.array([1., 0.]), None, np.array([0.8, 0.6])], reference)
        self.assertEqual(scores["samples"], 3)
        self.assertEqual(scores["detected"], 2)
        self.assertAlmostEqual(scores["coverage"], 2 / 3)
        self.assertEqual(scores["reference_cosine"][1], None)
        self.assertAlmostEqual(scores["reference_cosine"][2], 0.8)

    def test_zero_embedding_is_invalid(self):
        with self.assertRaises(ValueError):
            cosine(np.array([0., 0.]), np.array([1., 0.]))

    def test_multiple_faces_are_marked_ambiguous(self):
        class Analyzer:
            def get(self, _frame):
                return [SimpleNamespace(det_score=.9, embedding=np.array([1., 0.])),
                        SimpleNamespace(det_score=.8, embedding=np.array([0., 1.]))]

        embedding, score, count = detect_single_face_embedding(Analyzer(),
                                                               np.zeros((2, 2, 3)))
        self.assertIsNone(embedding)
        self.assertIsNone(score)
        self.assertEqual(count, 2)

    def test_late_detection_failure_blocks_drift_comparison(self):
        reference = np.array([1., 0.])
        result = summarize_embeddings([reference, reference, None], reference)
        self.assertEqual(result["early_coverage"], 1.)
        self.assertEqual(result["late_coverage"], 0.)
        self.assertFalse(result["score_usable"])
        self.assertIsNone(result["late_mean_reference_cosine"])


if __name__ == "__main__":
    unittest.main()

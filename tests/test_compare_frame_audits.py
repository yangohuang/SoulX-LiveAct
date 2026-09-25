import unittest

from compare_frame_audits import compare


def audit(audio, noise, latent, decoded, frames):
    return {"seed": 43, "fps": 24,
            "blocks": [{"audio_window_sha256": a,
                        "initial_noise_sha256": n,
                        "final_latent_sha256": l,
                        "decoded_block_sha256": d}
                       for a, n, l, d in zip(audio, noise, latent, decoded)],
            "pre_encoder_frame_sha256": frames}


class CompareFrameAuditsTests(unittest.TestCase):
    def test_localizes_audio_before_output_with_unchanged_noise(self):
        short = audit(["a", "b"], ["n", "n"], ["l", "x"], ["d", "x"],
                      ["f", "f", "x"])
        long = audit(["a", "c"], ["n", "n"], ["l", "y"], ["d", "y"],
                     ["f", "f", "y", "extra"])
        result = compare(short, long, {"source_embedding_sha256": "same"},
                         {"source_embedding_sha256": "same"})
        self.assertEqual(result["first_different_block_by_field"], {
            "audio_window_sha256": 1, "initial_noise_sha256": None,
            "final_latent_sha256": 1, "decoded_block_sha256": 1})
        self.assertEqual(result["first_different_pre_encoder_frame"], 2)

    def test_rejects_different_canonical_features(self):
        same = audit(["a"], ["n"], ["l"], ["d"], ["f"])
        with self.assertRaisesRegex(ValueError, "canonical"):
            compare(same, same, {"source_embedding_sha256": "x"},
                    {"source_embedding_sha256": "y"})


if __name__ == "__main__":
    unittest.main()

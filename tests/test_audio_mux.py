import subprocess
import tempfile
import unittest
from pathlib import Path

from util_liveact import add_audio_to_video
from video_output import VideoBlockWriter
import torch


class AudioMuxTests(unittest.TestCase):
    def test_invalid_audio_propagates_ffmpeg_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            silent_video = directory / "silent.mp4"
            invalid_audio = directory / "invalid.wav"
            invalid_audio.write_bytes(b"not an audio stream")
            with VideoBlockWriter(silent_video, fps=24) as writer:
                writer.append(torch.zeros(1, 3, 2, 16, 16, dtype=torch.bfloat16))
            with self.assertRaises(subprocess.CalledProcessError):
                add_audio_to_video(str(silent_video), str(invalid_audio),
                                   str(directory / "output.mp4"))


if __name__ == "__main__":
    unittest.main()

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from app.video import add_timestamp_watermark, timestamp_label, write_mp4


def test_timestamp_watermark_and_mp4(tmp_path: Path) -> None:
    timestamp = datetime(2026, 7, 15, 12, 34, 56)
    blank = np.zeros((180, 320, 3), dtype=np.uint8)
    stamped = add_timestamp_watermark(blank, timestamp)

    assert timestamp_label(timestamp) == "2026-07-15 12:34:56"
    assert np.any(stamped[:80, 120:] != 0)
    assert np.all(stamped[120:, :100] == 0)

    path = tmp_path / "preview.mp4"
    write_mp4(path, [stamped, stamped], fps=7)
    encoded = path.read_bytes()
    assert b"avc1" in encoded  # H.264 sample entry used by browser MP4 players.
    assert 0 <= encoded.find(b"moov") < encoded.find(b"mdat")  # faststart metadata first.
    assert path.with_name(f"{path.name}.browser-ready").exists()
    capture = cv2.VideoCapture(str(path))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 2
        assert round(capture.get(cv2.CAP_PROP_FPS)) == 7
    finally:
        capture.release()


def test_streaming_video_cancellation_during_encoding_cleans_subprocess(tmp_path, monkeypatch):
    import pytest
    from app import video
    calls = []
    class Process:
        def __init__(self, command, **kwargs):
            self.returncode = None
            Path(command[-1]).write_bytes(b"partial")
        def poll(self):
            return self.returncode
        def kill(self):
            calls.append("kill")
            self.returncode = -9
        def communicate(self, **kwargs):
            calls.append("reap")
            return "", ""
    monkeypatch.setattr(video, "_ffmpeg_executable", lambda: "ffmpeg")
    monkeypatch.setattr(video.subprocess, "Popen", Process)
    path = tmp_path / "cancel.mp4"
    path.write_bytes(b"source")
    with pytest.raises(ValueError, match="abgebrochen"):
        video.finalize_browser_mp4(path, cancelled=lambda: True)
    assert calls == ["kill", "reap"]
    assert not path.with_name("cancel.browser-tmp.mp4").exists()
    assert not path.with_name("cancel.mp4.browser-ready").exists()


def test_optional_watermark_label_preserves_default_and_supports_utc(monkeypatch):
    from app import video
    labels = []
    original = video.ImageDraw.ImageDraw.text
    def record_text(self, xy, text, *args, **kwargs):
        labels.append(text)
        return original(self, xy, text, *args, **kwargs)
    monkeypatch.setattr(video.ImageDraw.ImageDraw, 'text', record_text)
    timestamp = datetime(2025, 9, 15, 20, 45)
    blank = np.zeros((180, 320, 3), dtype=np.uint8)
    video.add_timestamp_watermark(blank, timestamp)
    stamped = video.add_timestamp_watermark(blank, timestamp, label='2025-09-15 18:45:00 (UTC)')
    assert labels == ['2025-09-15 20:45:00', '2025-09-15 18:45:00 (UTC)']
    assert np.any(stamped[:50] != 0)

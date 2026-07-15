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

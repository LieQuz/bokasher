import numpy as np

from bokasher.blur import apply_face_blur
from bokasher.types import FaceBox


def test_blur_without_boxes_returns_same_array() -> None:
    frame = np.full((40, 40, 3), 80, dtype=np.uint8)
    out = apply_face_blur(frame, [], strength=0.8)
    assert out is frame


def test_blur_softens_face_region() -> None:
    frame = np.zeros((80, 80, 3), dtype=np.uint8)
    frame[:, ::4] = 255
    out = apply_face_blur(frame, [FaceBox(20, 20, 40, 40)], strength=1.0)
    assert int(out[2, 2].mean()) == 0
    assert not np.array_equal(out[20:60, 20:60], frame[20:60, 20:60])
    # 縞が潰れて中心付近の分散が下がる
    assert float(out[30:50, 30:50].std()) < float(frame[30:50, 30:50].std())


def test_blur_skips_edge_sliver_without_crashing() -> None:
    # 4K 縦動画で顔が画面端に張り付くと、片側だけ数画素の ROI になる
    frame = np.full((3840, 2160, 3), 40, dtype=np.uint8)
    frame[:, ::8] = 220
    out = apply_face_blur(
        frame,
        [FaceBox(0, 800, 2160, 3), FaceBox(2140, 0, 8, 2000)],
        strength=0.8,
    )
    assert out.shape == frame.shape

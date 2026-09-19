from __future__ import annotations

import urllib.request

import cv2
import numpy as np
import pytest

from bokasher.detector import ensure_model

PORTRAIT_URL = "https://storage.googleapis.com/mediapipe-assets/portrait.jpg"


@pytest.fixture(scope="session")
def portrait_bgr(tmp_path_factory: pytest.TempPathFactory) -> np.ndarray:
    ensure_model()
    dest = tmp_path_factory.mktemp("media") / "portrait.jpg"
    urllib.request.urlretrieve(PORTRAIT_URL, dest)
    image = cv2.imread(str(dest))
    if image is None:
        pytest.skip("サンプル顔画像を読み込めませんでした")
    return image

import importlib.util
from pathlib import Path

import numpy as np

from eq.labelstudio.brush_rle import encode_rle, mask2rle


def _load_upstream_brush_module():
    path = Path('/opt/homebrew/lib/python3.10/site-packages/label_studio_converter/brush.py')
    spec = importlib.util.spec_from_file_location('ls_upstream_brush', path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_encode_rle_matches_upstream_converter_padding_contract():
    upstream = _load_upstream_brush_module()
    arr = np.array([0, 0, 0, 255, 255, 255, 0, 0], dtype=np.uint8)

    assert encode_rle(arr) == upstream.encode_rle(arr)


def test_mask2rle_matches_upstream_converter_for_disconnected_mask():
    upstream = _load_upstream_brush_module()
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[2:10, 3:12] = 255
    mask[18:28, 20:30] = 255

    assert mask2rle(mask) == upstream.mask2rle(mask)

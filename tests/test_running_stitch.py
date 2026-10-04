from unittest.mock import patch

import pytest

from lib.stitches.running_stitch import split_segment_even_n


def test_split_segment_even_n_uses_cpu_when_requested(monkeypatch):
    monkeypatch.setenv("INKSTITCH_COMPUTE_BACKEND", "cpu")
    with patch("lib.utils.webgpu._get_resources", side_effect=AssertionError("WebGPU should not initialize")):
        points = split_segment_even_n((0, 0), (10, 0), 5)

    assert [point.x for point in points] == pytest.approx([2, 4, 6, 8])
    assert [point.y for point in points] == pytest.approx([0, 0, 0, 0])


def test_split_segment_even_n_falls_back_without_adapter(monkeypatch):
    monkeypatch.setenv("INKSTITCH_COMPUTE_BACKEND", "webgpu")
    with patch("lib.utils.webgpu._get_resources", return_value=None):
        points = split_segment_even_n((0, 0), (10, 0), 5)

    assert [point.x for point in points] == pytest.approx([2, 4, 6, 8])
    assert [point.y for point in points] == pytest.approx([0, 0, 0, 0])


def test_split_segment_even_n_uses_threaded_cpu(monkeypatch):
    monkeypatch.setenv("INKSTITCH_COMPUTE_BACKEND", "threaded")
    with patch("lib.utils.webgpu._get_resources", side_effect=AssertionError("WebGPU should not initialize in threaded mode")):
        points = split_segment_even_n((0, 0), (10, 0), 300)

    assert len(points) == 299
    assert [point.x for point in points[:3]] == pytest.approx([10 / 300, 20 / 300, 30 / 300])
    assert [point.y for point in points[:3]] == pytest.approx([0, 0, 0])
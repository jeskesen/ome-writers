from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from ome_writers._schema import AcquisitionSettings, Dimension, OmeZarrFormat
from ome_writers._stream import create_stream
from tests._utils import read_array_data

pytest.importorskip("acquire_zarr", reason="acquire-zarr not installed")


def test_acquire_zarr_full_streaming_support(tmp_path: Path) -> None:
    """Test that our backend abstraction doesn't break cool acquire-zarr features.

    One very nice thing about acquire-zarr's stream is that it makes no assumptions
    about shape of each buffer being passed to `stream.append()`,  C-contiguous
    buffers are simply concatenated according to the dimensionality declared in
    the settings.

    This test ensures that our backend abstraction preserves this behavior.
    """

    settings = AcquisitionSettings(
        root_path=str(tmp_path / "output.zarr"),
        dimensions=[
            Dimension(name="z", count=18, chunk_size=6, unit="um", scale=0.5),
            Dimension(name="y", count=128, chunk_size=64, unit="um", scale=0.1),
            Dimension(name="x", count=128, chunk_size=64, unit="um", scale=0.1),
        ],
        dtype="uint16",
        format="acquire-zarr",
    )

    shape = tuple(d.count or 1 for d in settings.dimensions)
    flat_data = np.arange(np.prod(shape), dtype=settings.dtype)
    # break the data into 10 arbitrary, non-frame/chunk-aligned, somewhat random pieces
    boundaries = [0, 1500, 3000, 5000, 7000, 9000, 12000, 15000, 18000, 22000, None]
    append_bits = [flat_data[start:stop] for start, stop in pairwise(boundaries)]

    with create_stream(settings) as stream:
        for bit in append_bits:
            stream.append(bit)

    output_data = read_array_data(settings.output_path)
    assert output_data.shape == (18, 128, 128)
    assert output_data.dtype == np.dtype(settings.dtype)
    assert np.array_equal(output_data.flatten(), flat_data)


def _capture_stream_settings(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Spy on ZarrStream construction, recording the StreamSettings.max_threads."""
    import acquire_zarr as az

    from ome_writers._backends import _acquire_zarr

    captured: dict[str, int] = {}
    real_zarr_stream = az.ZarrStream

    def spy(stream_settings: az.StreamSettings) -> az.ZarrStream:
        captured["max_threads"] = stream_settings.max_threads
        return real_zarr_stream(stream_settings)

    monkeypatch.setattr(_acquire_zarr.az, "ZarrStream", spy)
    return captured


def _run_tiny_stream(settings: AcquisitionSettings) -> None:
    shape = tuple(d.count or 1 for d in settings.dimensions)
    flat_data = np.arange(np.prod(shape), dtype=settings.dtype)
    with create_stream(settings) as stream:
        stream.append(flat_data)


def test_acquire_zarr_max_threads_passthrough(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`OmeZarrFormat.max_threads` is applied to the acquire-zarr stream pool."""
    captured = _capture_stream_settings(monkeypatch)
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "output.zarr"),
        dimensions=[
            Dimension(name="z", count=4, chunk_size=2),
            Dimension(name="y", count=8, chunk_size=8),
            Dimension(name="x", count=8, chunk_size=8),
        ],
        dtype="uint16",
        format=OmeZarrFormat(backend="acquire-zarr", max_threads=2),
    )
    _run_tiny_stream(settings)
    assert captured["max_threads"] == 2


def test_acquire_zarr_max_threads_default_untouched(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When unset, the acquire-zarr default thread pool is left in place."""
    import acquire_zarr as az

    captured = _capture_stream_settings(monkeypatch)
    settings = AcquisitionSettings(
        root_path=str(tmp_path / "output.zarr"),
        dimensions=[
            Dimension(name="z", count=4, chunk_size=2),
            Dimension(name="y", count=8, chunk_size=8),
            Dimension(name="x", count=8, chunk_size=8),
        ],
        dtype="uint16",
        format=OmeZarrFormat(backend="acquire-zarr"),
    )
    _run_tiny_stream(settings)
    assert captured["max_threads"] == az.StreamSettings().max_threads


def test_ome_zarr_format_max_threads_must_be_positive() -> None:
    """`max_threads` must be a positive integer when supplied."""
    with pytest.raises(ValueError):
        OmeZarrFormat(max_threads=0)

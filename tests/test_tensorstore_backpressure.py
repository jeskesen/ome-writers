"""In-flight write-future cap for the tensorstore backend (#214).

tensorstore's `array[index].write()` is asynchronous; each returned future pins
its source frame in RAM until the write commits. Without a cap the future queue
grows worker memory linearly with frames written even when storage keeps up
(the #212 leak). These tests pin the bounded-queue behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from ome_writers._schema import AcquisitionSettings, Dimension, OmeZarrFormat
from ome_writers._stream import create_stream
from tests._utils import read_array_data

pytest.importorskip("tensorstore", reason="tensorstore not installed")

from typing import TYPE_CHECKING

from ome_writers._backends._tensorstore import (
    _DEFAULT_MAX_INFLIGHT,
    TensorstoreBackend,
)

if TYPE_CHECKING:
    from pathlib import Path


def _settings(
    tmp_path: Path, *, n_frames: int, max_inflight: int | None
) -> AcquisitionSettings:
    return AcquisitionSettings(
        root_path=str(tmp_path / "output.zarr"),
        dimensions=[
            # chunk_size=1 on the index dim disables chunk buffering, so each
            # appended frame becomes exactly one async write (one future).
            Dimension(name="t", count=n_frames, chunk_size=1),
            Dimension(name="y", count=8, chunk_size=8),
            Dimension(name="x", count=8, chunk_size=8),
        ],
        dtype="uint16",
        format=OmeZarrFormat(backend="tensorstore", max_inflight_writes=max_inflight),
    )


def test_inflight_cap_bounds_future_queue(tmp_path: Path) -> None:
    """The future queue never exceeds the cap, and no frame is lost."""
    n_frames = 200
    cap = 4
    settings = _settings(tmp_path, n_frames=n_frames, max_inflight=cap)

    with create_stream(settings) as stream:
        backend = stream._backend
        assert isinstance(backend, TensorstoreBackend)
        assert backend._max_inflight == cap
        for i in range(n_frames):
            stream.append(np.full((8, 8), i, dtype="uint16"))
            # After every write the queue is drained back to <= cap.
            assert len(backend._futures) <= cap

    output = read_array_data(settings.output_path)
    assert output.shape == (n_frames, 8, 8)
    for i in range(n_frames):
        assert np.array_equal(output[i], np.full((8, 8), i, dtype="uint16"))


def test_inflight_cap_defaults_when_unset(tmp_path: Path) -> None:
    """Leaving `max_inflight_writes` unset uses the backend default."""
    settings = _settings(tmp_path, n_frames=4, max_inflight=None)
    with create_stream(settings) as stream:
        backend = stream._backend
        assert isinstance(backend, TensorstoreBackend)
        assert backend._max_inflight == _DEFAULT_MAX_INFLIGHT


def test_inflight_cap_is_configurable(tmp_path: Path) -> None:
    """An explicit `max_inflight_writes` is honored by the backend."""
    settings = _settings(tmp_path, n_frames=4, max_inflight=2)
    with create_stream(settings) as stream:
        assert stream._backend._max_inflight == 2


def test_max_inflight_writes_must_be_positive() -> None:
    """`max_inflight_writes` must be a positive integer when supplied."""
    with pytest.raises(ValueError):
        OmeZarrFormat(max_inflight_writes=0)

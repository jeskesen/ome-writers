from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Any, Literal

from ome_writers._backends._yaozarrs import YaozarrsBackend

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np
    from tensorstore import Future

    from ome_writers._router import FrameRouter
    from ome_writers._schema import AcquisitionSettings

# Default cap on outstanding (not-yet-committed) async write futures. A writer
# that is keeping up never reaches this; it is the RAM backstop that converts
# unbounded future growth into producer backpressure when storage falls behind
# (#214 -- the #212 leak grew worker RSS ~81 GB -> ~171 GB across one capture
# because every write future pinned its source frame until finalize()).
_DEFAULT_MAX_INFLIGHT = 64


class TensorstoreBackend(YaozarrsBackend):
    """OME-Zarr writer using tensorstore via yaozarrs."""

    def _get_yaozarrs_writer(self) -> Literal["tensorstore"]:
        return "tensorstore"

    def __init__(self) -> None:
        super().__init__()
        self._futures: deque[Future] = deque()
        self._max_inflight: int = _DEFAULT_MAX_INFLIGHT

    def prepare(self, settings: AcquisitionSettings, router: FrameRouter) -> None:
        """Capture the in-flight write cap, then build the OME-Zarr structure."""
        max_inflight = getattr(settings.format, "max_inflight_writes", None)
        self._max_inflight = (
            max_inflight if max_inflight is not None else _DEFAULT_MAX_INFLIGHT
        )
        super().prepare(settings, router)

    def _write(self, array: Any, index: tuple[int, ...], frame: np.ndarray) -> None:
        """Write frame to array at specified index, async for tensorstore.

        tensorstore's write is asynchronous and the returned future pins its
        source frame in RAM until the write commits. Cap the number of
        outstanding futures: once the queue exceeds ``_max_inflight``, drain the
        oldest (block on its commit) before enqueuing more, so memory stays
        bounded regardless of how many frames are written.
        """
        self._futures.append(array[index].write(frame))
        while len(self._futures) > self._max_inflight:
            self._futures.popleft().result()

    def _resize(self, array: Any, new_shape: Sequence[int]) -> None:
        """Resize array to new shape, using exclusive_max for tensorstore."""
        array.resize(exclusive_max=new_shape).result()

    def finalize(self) -> None:
        """Flush and release resources."""
        while self._futures:
            self._futures.popleft().result()
        super().finalize()

"""Mouse handler for interactive bounding box drawing."""

from dataclasses import dataclass

import cv2


@dataclass(frozen=True)
class BoundingBox:
    """Represent an axis-aligned bounding box.

    Attributes:
        x_min (int): Left edge coordinate.
        y_min (int): Top edge coordinate.
        x_max (int): Right edge coordinate.
        y_max (int): Bottom edge coordinate.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int


class MouseHandler:
    """Track mouse drag events to create bounding boxes.

    Attributes:
        boxes (list[BoundingBox]): Finalized bounding boxes.
    """

    boxes: list[BoundingBox]

    def __init__(self) -> None:
        self.boxes: list[BoundingBox] = []
        self._current_start: tuple[int, int] | None = None
        self._current_end: tuple[int, int] | None = None

    @property
    def temp_box(self) -> BoundingBox | None:
        """Return the in-progress bounding box, or None if not dragging.

        Returns:
            BoundingBox | None: The temporary box, or ``None``.
        """
        if self._current_start is None or self._current_end is None:
            return None
        return self._create_box(
            self._current_start[0],
            self._current_start[1],
            self._current_end[0],
            self._current_end[1],
        )

    def handle_event(
        self,
        event: int,
        x: int,
        y: int,
        flags: int,
        param: object,
    ) -> None:
        """Handle an OpenCV mouse callback event.

        Args:
            event (int): OpenCV mouse event type.
            x (int): Horizontal pixel coordinate.
            y (int): Vertical pixel coordinate.
            flags (int): OpenCV event flags.
            param (object): User-defined data passed by OpenCV.

        Raises:
            RuntimeError: If LBUTTONUP arrives without a prior LBUTTONDOWN.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_start = (x, y)
            self._current_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._current_start is not None:
                self._current_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self._current_start is None:
                raise RuntimeError("LBUTTONUP received without a preceding LBUTTONDOWN")
            new_box = self._create_box(
                self._current_start[0], self._current_start[1], x, y
            )
            if new_box is not None:
                self.boxes.append(new_box)
            self._current_start = None
            self._current_end = None

    def reset(self) -> None:
        """Clear all stored boxes and reset drag state."""
        self.boxes = []
        self._current_start = None
        self._current_end = None

    def _create_box(self, x1: int, y1: int, x2: int, y2: int) -> BoundingBox | None:
        """Create a normalized bounding box from two corner points.

        Args:
            x1 (int): First corner x coordinate.
            y1 (int): First corner y coordinate.
            x2 (int): Second corner x coordinate.
            y2 (int): Second corner y coordinate.

        Returns:
            BoundingBox | None: The bounding box, or ``None`` if too small.
        """
        if abs(x1 - x2) <= 1 or abs(y1 - y2) <= 1:
            return None
        return BoundingBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

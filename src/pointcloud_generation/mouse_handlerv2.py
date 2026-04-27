import cv2
from dataclasses import dataclass

@dataclass(frozen=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int


class MouseHandler:
    def __init__(self):
        self.boxes: list[BoundingBox] = []
        self._current_start: tuple | None = None
        self._current_end: tuple | None = None
    
    @property
    def temp_box(self) -> BoundingBox | None:
        if self._current_start is None or self._current_end is None:
            return None
        return self._create_box(self._current_start[0], self._current_start[1], self._current_end[0], self._current_end[1])

    def handle_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._current_start = (x, y)
            self._current_end = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._current_start is not None:
                self._current_end = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            if self._current_start is None:
                raise RuntimeError("LBUTTONUP received without a preceding LBUTTONDOWN")
            new_box = self._create_box(self._current_start[0], self._current_start[1], x, y)
            if new_box is not None:
                self.boxes.append(new_box)
            self._current_start = None
            self._current_end = None

    def reset(self):
        self.boxes = []
        self._current_start = None
        self._current_end = None

    def _create_box(self, x1, y1, x2, y2) -> BoundingBox | None:
        if abs(x1 - x2) <= 1 or abs(y1 - y2) <= 1:
            return None
        return BoundingBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

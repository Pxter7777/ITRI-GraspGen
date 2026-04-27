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
    def num_boxes(self) -> int:
        return len(self.boxes)
    
    @property
    def temp_box(self) -> BoundingBox | None:
        if self._current_start is None or self._current_end is None:
            return None
        return BoundingBox(x_min=self._current_start[0], y_min=self._current_start[1], x_max=self._current_end[0], y_max=self._current_end[1])

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
            sx, sy = self._current_start
            if abs(sx - x) > 1 and abs(sy - y) > 1:
                self.boxes.append(BoundingBox(x_min=min(sx, x), y_min=min(sy, y), x_max=max(sx, x), y_max=max(sy, y)))
            self._current_start = None
            self._current_end = None

    def reset(self):
        self.boxes = []
        self._current_start = None
        self._current_end = None


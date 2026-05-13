"""Test the MouseHandler bounding-box drawing logic."""

import cv2
import pytest

from pointcloud_generation.mouse_handlerv2 import BoundingBox, MouseHandler


@pytest.fixture
def handler() -> MouseHandler:
    """Create a fresh MouseHandler instance.

    Returns:
        MouseHandler: A new handler.
    """
    return MouseHandler()


def draw_box(handler: MouseHandler, x1: int, y1: int, x2: int, y2: int) -> None:
    """Simulate a complete mouse drag from (x1,y1) to (x2,y2).

    Args:
        handler (MouseHandler): The handler to receive events.
        x1 (int): Start x coordinate.
        y1 (int): Start y coordinate.
        x2 (int): End x coordinate.
        y2 (int): End y coordinate.
    """
    handler.handle_event(cv2.EVENT_LBUTTONDOWN, x1, y1, 0, None)
    handler.handle_event(cv2.EVENT_MOUSEMOVE, x2, y2, 0, None)
    handler.handle_event(cv2.EVENT_LBUTTONUP, x2, y2, 0, None)


def test_single_box(handler: MouseHandler):
    """Record one box from a simple drag.

    Args:
        handler (MouseHandler): Pytest fixture providing a fresh handler.
    """
    draw_box(handler, 10, 20, 50, 60)
    assert handler.boxes == [BoundingBox(10, 20, 50, 60)]


def test_box_coordinates_normalized(handler: MouseHandler):
    """Drawn right-to-left and bottom-to-top -- get_boxes should still return min/max.

    Args:
        handler (MouseHandler): Pytest fixture providing a fresh handler.
    """
    draw_box(handler, 50, 60, 10, 20)
    assert handler.boxes == [BoundingBox(10, 20, 50, 60)]


def test_multiple_boxes(handler: MouseHandler):
    """Record two boxes from consecutive drags.

    Args:
        handler (MouseHandler): Pytest fixture providing a fresh handler.
    """
    draw_box(handler, 0, 0, 30, 30)
    draw_box(handler, 50, 50, 100, 100)
    assert handler.boxes == [BoundingBox(0, 0, 30, 30), BoundingBox(50, 50, 100, 100)]


def test_box_too_small_is_discarded(handler: MouseHandler):
    """Discard a box that is too small to be meaningful.

    Args:
        handler (MouseHandler): Pytest fixture providing a fresh handler.
    """
    draw_box(handler, 10, 10, 11, 10)  # width=1, height=0 — below threshold
    assert handler.boxes == []


def test_mousemove_updates_current_end_while_drawing(handler: MouseHandler):
    """Track the latest mouse position during an active drag.

    Args:
        handler (MouseHandler): Pytest fixture providing a fresh handler.
    """
    handler.handle_event(cv2.EVENT_LBUTTONDOWN, 0, 0, 0, None)
    handler.handle_event(cv2.EVENT_MOUSEMOVE, 25, 25, 0, None)
    handler.handle_event(cv2.EVENT_MOUSEMOVE, 50, 50, 0, None)
    assert handler._current_end == (50, 50)  # type: ignore[reportPrivateUsage]
    handler.handle_event(cv2.EVENT_LBUTTONUP, 50, 50, 0, None)
    assert handler.boxes == [BoundingBox(0, 0, 50, 50)]


def test_reset_clears_all_state(handler: MouseHandler):
    """Clear all boxes and drawing state on reset.

    Args:
        handler (MouseHandler): Pytest fixture providing a fresh handler.
    """
    draw_box(handler, 0, 0, 50, 50)
    handler.reset()
    assert handler.boxes == []
    assert handler._current_start is None  # type: ignore[reportPrivateUsage]
    assert handler._current_end is None  # type: ignore[reportPrivateUsage]


def test_mousemove_ignored_when_not_drawing(handler: MouseHandler):
    """Ignore mouse-move events when no drag is in progress.

    Args:
        handler (MouseHandler): Pytest fixture providing a fresh handler.
    """
    handler.handle_event(cv2.EVENT_MOUSEMOVE, 99, 99, 0, None)
    assert handler._current_end is None  # type: ignore[reportPrivateUsage]


def test_lbuttonup_without_lbuttondown_raises(handler: MouseHandler):
    """Raise RuntimeError on button-up without a preceding button-down.

    Args:
        handler (MouseHandler): Pytest fixture providing a fresh handler.
    """
    with pytest.raises(RuntimeError):
        handler.handle_event(cv2.EVENT_LBUTTONUP, 50, 50, 0, None)


def test_temp_box_during_draw(handler: MouseHandler):
    """Expose a temporary box while drawing and clear it on release.

    Args:
        handler (MouseHandler): Pytest fixture providing a fresh handler.
    """
    assert handler.temp_box is None
    handler.handle_event(cv2.EVENT_LBUTTONDOWN, 0, 0, 0, None)
    handler.handle_event(cv2.EVENT_MOUSEMOVE, 30, 40, 0, None)
    assert handler.temp_box == BoundingBox(0, 0, 30, 40)
    handler.handle_event(cv2.EVENT_LBUTTONUP, 30, 40, 0, None)
    assert handler.temp_box is None


def test_temp_box_during_draw_normallized(handler: MouseHandler):
    """Drawn right-to-left and bottom-to-top -- get_boxes should still return min/max.

    Args:
        handler (MouseHandler): Pytest fixture providing a fresh handler.
    """
    handler.handle_event(cv2.EVENT_LBUTTONDOWN, 50, 60, 0, None)
    handler.handle_event(cv2.EVENT_MOUSEMOVE, 10, 20, 0, None)
    assert handler.temp_box == BoundingBox(10, 20, 50, 60)

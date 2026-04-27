import cv2
import pytest
from pointcloud_generation.mouse_handlerv2 import MouseHandler


@pytest.fixture
def handler():
    return MouseHandler()


def draw_box(handler, x1, y1, x2, y2):
    """Simulate a complete mouse drag from (x1,y1) to (x2,y2)."""
    handler.select_box(cv2.EVENT_LBUTTONDOWN, x1, y1, 0, None)
    handler.select_box(cv2.EVENT_MOUSEMOVE, x2, y2, 0, None)
    handler.select_box(cv2.EVENT_LBUTTONUP, x2, y2, 0, None)


def test_single_box(handler):
    draw_box(handler, 10, 20, 50, 60)
    assert handler.num_boxes == 1
    assert handler.get_boxes() == [[10, 20, 50, 60]]


def test_box_coordinates_normalized(handler):
    """Drawn right-to-left and bottom-to-top — get_boxes should still return min/max."""
    draw_box(handler, 50, 60, 10, 20)
    assert handler.get_boxes() == [[10, 20, 50, 60]]


def test_multiple_boxes(handler):
    draw_box(handler, 0, 0, 30, 30)
    draw_box(handler, 50, 50, 100, 100)
    assert handler.num_boxes == 2
    assert handler.get_boxes() == [[0, 0, 30, 30], [50, 50, 100, 100]]


def test_box_too_small_is_discarded(handler):
    draw_box(handler, 10, 10, 11, 10)  # width=1, height=0 — below threshold
    assert handler.num_boxes == 0
    assert handler.get_boxes() == []


def test_mousemove_updates_end_point_while_drawing(handler):
    handler.select_box(cv2.EVENT_LBUTTONDOWN, 0, 0, 0, None)
    handler.select_box(cv2.EVENT_MOUSEMOVE, 25, 25, 0, None)
    handler.select_box(cv2.EVENT_MOUSEMOVE, 50, 50, 0, None)
    assert handler.box_end_points[-1] == (50, 50)
    handler.select_box(cv2.EVENT_LBUTTONUP, 50, 50, 0, None)
    assert handler.get_boxes() == [[0, 0, 50, 50]]


def test_reset_clears_all_state(handler):
    draw_box(handler, 0, 0, 50, 50)
    handler.reset()
    assert handler.num_boxes == 0
    assert handler.box_start_points == []
    assert handler.box_end_points == []
    assert handler.drawing_box is False


def test_mousemove_ignored_when_not_drawing(handler):
    handler.select_box(cv2.EVENT_MOUSEMOVE, 99, 99, 0, None)
    assert handler.box_end_points == []

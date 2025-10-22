import cv2

class MouseHandler:
    def __init__(self):
        self.drawing_box = False
        self.box_start_points = []
        self.box_end_points = []
        self.num_boxes = 0
    def select_box(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing_box = True
            self.num_boxes += 1
            self.box_start_points.append((x, y))
            self.box_end_points.append((x, y))
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing_box:
                self.box_end_points[-1] = (x,y)
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing_box = False
            if abs(self.box_start_points[-1][0] - x) > 1 and abs(self.box_start_points[-1][1] - y) > 1:
                self.box_end_points[-1] = (x, y)
            else:
                # If the box is too small, reset it
                self.num_boxes -= 1
                self.box_start_points.pop()
                self.box_end_points.pop()
    def reset(self):
        self.drawing_box = False
        self.box_start_points = []
        self.box_end_points = []
        self.num_boxes = 0
    def get_boxes(self):
        boxes = []
        for i in range(self.num_boxes):
            boxes.append([
                min(self.box_start_points[i][0], self.box_end_points[i][0]),
                min(self.box_start_points[i][1], self.box_end_points[i][1]),
                max(self.box_start_points[i][0], self.box_end_points[i][0]),
                max(self.box_start_points[i][1], self.box_end_points[i][1])
            ])
        #x1 = min(box_start_point[0], box_end_point[0])
        #y1 = min(box_start_point[1], box_end_point[1])
        #x2 = max(box_start_point[0], box_end_point[0])
        #y2 = max(box_start_point[1], box_end_point[1])
        return boxes

def select_box(event, x, y, flags, param):
    """Mouse callback function to select a bounding box."""
    global drawing_box, box_start_point, box_end_point, box_defined

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_box = True
        box_defined = False
        box_start_point = (x, y)
        box_end_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_box:
            box_end_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing_box = False
        # Ensure the box has a non-zero area by checking width and height
        if abs(box_start_point[0] - x) > 1 and abs(box_start_point[1] - y) > 1:
            box_end_point = (x, y)
            box_defined = True
        else:
            # If the box is too small, reset it
            box_start_point = (-1, -1)
            box_end_point = (-1, -1)


def reset_box():
    global box_defined, drawing_box, box_start_point, box_end_point
    box_defined = False
    drawing_box = False
    box_start_point = (-1, -1)
    box_end_point = (-1, -1)


def get_box():
    global box_start_point, box_end_point
    x1 = min(box_start_point[0], box_end_point[0])
    y1 = min(box_start_point[1], box_end_point[1])
    x2 = max(box_start_point[0], box_end_point[0])
    y2 = max(box_start_point[1], box_end_point[1])
    return [x1, y1, x2, y2]

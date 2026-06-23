import cv2
import numpy as np
import math

class DraggableObject:
    def __init__(self, x, y, w, h, color, label=""):
        self.x = x  # center x
        self.y = y  # center y
        self.w = w
        self.h = h
        self.color = color
        self.label = label
        self.selected = False
        self.grabbed = False

    def draw(self, frame):
        x1 = int(self.x - self.w / 2)
        y1 = int(self.y - self.h / 2)
        x2 = int(self.x + self.w / 2)
        y2 = int(self.y + self.h / 2)

        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), self.color, -1)
        alpha = 0.6 if not self.grabbed else 0.85
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        border_color = (255, 255, 255) if self.selected or self.grabbed else self.color
        thickness = 3 if self.selected or self.grabbed else 1
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, thickness)

        if self.label:
            cv2.putText(frame, self.label, (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    def is_point_inside(self, px, py):
        return (self.x - self.w / 2 < px < self.x + self.w / 2 and
                self.y - self.h / 2 < py < self.y + self.h / 2)

    def move_to(self, px, py):
        self.x = px
        self.y = py

    def resize(self, scale_factor):
        self.w = max(40, min(300, self.w * scale_factor))
        self.h = max(40, min(300, self.h * scale_factor))


class DrawingStroke:
    """Garis yang digambar dan bisa dipindahkan"""
    def __init__(self, points, color=(0, 200, 255)):
        self.points = points  # list of (x, y)
        self.color = color
        self.grabbed = False
        self.offset = (0, 0)
    
    def draw(self, frame):
        """Gambar garis pada frame"""
        if len(self.points) >= 2:
            for i in range(len(self.points) - 1):
                thickness = 5 if self.grabbed else 4
                cv2.line(frame, self.points[i], self.points[i+1], self.color, thickness)
    
    def get_bounds(self):
        """Dapatkan bounding box dari garis"""
        if not self.points:
            return None
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return min(xs), min(ys), max(xs), max(ys)
    
    def is_point_inside(self, px, py):
        """Cek apakah point dekat dengan garis (within 15 pixels)"""
        for point in self.points:
            dist = math.hypot(px - point[0], py - point[1])
            if dist < 15:
                return True
        return False
    
    def move_to(self, px, py):
        """Pindahkan garis ke posisi baru"""
        if self.points:
            old_x, old_y = self.points[0]
            dx = px - old_x - self.offset[0]
            dy = py - old_y - self.offset[1]
            self.points = [(x + dx, y + dy) for x, y in self.points]
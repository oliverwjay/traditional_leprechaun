import cv2
import numpy as np
from enum import Enum


class InteractionMode(Enum):
    COMPOSITE = 1
    TEACH_COLOR = 2
    TEACH_OBJECT = 3


class DetectionController:
    def __init__(self):
        """
        Builds detection controller to handle detection between the models and the UI
        """
        self.bgr_frame = None  # Raw blue, green, and red
        self.hsv_frame = None  # Raw hue, saturation, and value
        self.processed_frame = None  # Processed output frame
        self.vc = cv2.VideoCapture(0)  # Camera

        self.mode = InteractionMode.COMPOSITE  # Track how the user is interacting
        self.selected_component = None  # Track the current

        self.vc.set(3, 640)  # set width
        self.vc.set(4, 360)  # set height

    def handle_click(self, x, y):
        """
        Handles clicking the image
        :param x: x coord of click
        :param y: y coord of click
        :return: None
        """
        print(f"Click at {(y, x)} with hsv value {self.hsv_frame[y, x]}"
              f"and processed value {self.processed_frame[y, x]}")

    def process_frame(self):
        """
        Pulls and processes the next frame
        :return: Filtered frame
        """
        ret, self.bgr_frame = self.vc.read()  # Read frame
        self.hsv_frame = cv2.cvtColor(self.bgr_frame, cv2.COLOR_BGR2HSV)  # Convert to HSV

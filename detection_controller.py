import cv2
import numpy as np
from enum import Enum
from visual_object import GenericObject
import time


class InteractionMode(Enum):
    COMPOSITE = 1
    TEACH_COLOR = 2
    TEACH_CONTOUR = 3
    TEACH_OBJECT = 4


class InputMode(Enum):
    NONE = 1
    CAMERA = 2
    FILE = 3
    STATIC = 4


class DetectionController:
    def __init__(self):
        """
        Builds detection controller to handle detection between the models and the UI
        """
        self.bgr_frame = None  # Raw blue, green, and red
        self.hsv_frame = None  # Raw hue, saturation, and value
        self.processed_frame = None  # Processed output frame
        self.vc = cv2.VideoCapture(0)  # Camera
        self.input_mode = InputMode.NONE
        self.interaction_mode = InteractionMode.TEACH_CONTOUR

        self.object = GenericObject()
        self.mode = InteractionMode.COMPOSITE  # Track how the user is interacting
        self.selected_component = None  # Track the current

        self.vc.set(3, 640)  # set width
        self.vc.set(4, 360)  # set height

    def handle_click(self, x, y):
        """
        Handles clicking the image
        :param x: x coord of click
        :param y: y coord of click
        :return: Text to output
        """
        self.object.components[self.selected_component].color.add_data(self.hsv_frame[y, x])
        self.object.components[self.selected_component].color.calculate_stats()
        return f"Click at {(y, x)} with hsv value {self.hsv_frame[y, x]}, " \
               f"bgr {self.bgr_frame[y, x]} and processed value {self.processed_frame[y, x]}"

    def save_contour(self, x, y):
        """
        Saves the first contour selected
        :return: None
        """
        if self.interaction_mode == InteractionMode.TEACH_CONTOUR:
            self.object.components[self.selected_component].define_contour(self.hsv_frame, x, y)
        elif self.interaction_mode == InteractionMode.TEACH_OBJECT:
            self.object.add_contour(x, y, self.selected_component)
            self.interaction_mode = InteractionMode.TEACH_CONTOUR

    def save_sizes(self):
        """
        Saves the expected sizes of the models
        :return: None
        """
        self.interaction_mode = InteractionMode.TEACH_OBJECT

    def clear_sizes(self):
        """
        Clears all the poses of the given component
        :return: None
        """
        self.object.components[self.selected_component].exp_poses = []
        self.object.components[self.selected_component].contour = None

    def set_slider(self, slider_name, new_size):
        """
        Changes the kernal size to open for a the selected component
        :param slider_name: Name of the slider to change
        :param new_size: The new size for the opening kernel
        :return: None
        """
        self.object.components[self.selected_component].color.slider_stats[slider_name] = new_size

    def get_slider_values(self):
        """
        Returns the size of the opening kernel for the selected kernel
        :return: size of the opening kernel for the selected kernel
        """
        return self.object.components[self.selected_component].color.slider_stats

    def update_image(self):
        """
        Pulls a frame and processes it
        :return: Raw and processed frames
        """
        if self.input_mode == InputMode.CAMERA:
            ret, raw = self.vc.read()  # Read frame
            return self.process_frame(raw)
        elif self.input_mode == InputMode.FILE or self.input_mode == InputMode.STATIC:
            return self.process_frame(self.bgr_frame)

    def process_from_file(self, filename):
        """
        Reads an image from file and processes it
        :param filename: Filename to read
        :return: raw and processed frames
        """
        frame = cv2.imread(filename)
        self.input_mode = InputMode.FILE
        return self.process_frame(frame)

    def set_input_to_camera(self):
        """
        Switches the input mode to camera
        :return: None
        """
        self.input_mode = InputMode.CAMERA

    def set_input_to_static(self):
        """
        Switches the input mode to static
        :return: None
        """
        self.input_mode = InputMode.STATIC

    def clear_color(self):
        """
        Resets all data for selected color
        :return: None
        """
        self.object.clear_component(self.selected_component)

    def process_frame(self, frame):
        """
        Pulls and processes the next frame
        :return: raw and processed frames
        """
        self.bgr_frame = cv2.resize(frame, (640, 360))
        self.hsv_frame = cv2.cvtColor(self.bgr_frame, cv2.COLOR_BGR2HSV)  # Convert to HSV
        self.processed_frame = self.object.components[self.selected_component].process_image(self.hsv_frame)

        # Find object
        for component in self.object.components.values():
            component.process_image(self.hsv_frame)
        with_leprechaun = self.object.find_object(self.bgr_frame)

        rgb_frame = cv2.cvtColor(with_leprechaun, cv2.COLOR_BGR2RGB)
        rgb_processed = cv2.cvtColor(self.processed_frame, cv2.COLOR_BGR2RGB)
        return rgb_frame, rgb_processed

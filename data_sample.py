import numpy as np
import cv2
import os
import pickle


class DataSample:
    """
    Handles a sample of data to evaluate against
    """

    def __init__(self, input_data=None):
        """
        Builds a sample
        :param input_data: File to get data from
        """
        print("Initializing!!!")
        self.data = []
        if input_data is not None:
            self.data = input_data
        self.sd = None
        self.mean = None
        self.data_file = input_data

        if input_data is not None and os.path.isfile(input_data):
            self.data = pickle.load(open(input_data, "rb"))
            self.calculate_stats()

    def calculate_stats(self):
        """
        Calculates mean and deviation of model
        :return: None
        """
        if self.data is None:
            return
        data_array = np.array(self.data)
        self.mean = np.mean(data_array, axis=0)
        self.sd = np.std(data_array, axis=0)

    def add_data(self, new_data_point):
        """
        Adds a data point to the model
        :param new_data_point: New data point
        :return:
        """
        # Check data size
        if len(self.data) > 0 and len(self.data[0]) != len(new_data_point):
            raise Exception("Data point size does not match sample")
        # Add to library
        self.data.append(new_data_point)

    def to_export(self):
        if self.data_file is not None:
            pickle.dump(self.data, open(self.data_file, "wb"))


def make_kernel(k_size, kernel=True):
    k_size |= 1
    k_dims = (k_size, k_size)
    if kernel:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_dims)
    else:
        return k_dims


class ColorSample(DataSample):
    def __init__(self, input_data=None):
        super().__init__(input_data)
        self.slider_stats = {'open': 0, 'close': 0, 'blur': 0, 'threshold': 50, 'contour threshold': 50}
        self.contour = None

    def binarize_image(self, image):
        if len(self.data) < 10:
            return None
        coef = 1 / (2 * np.pi * np.square(self.sd))
        diff_x_mu = image - self.mean
        pdf_exp = -np.square(diff_x_mu / self.sd) / 2
        pdf = np.exp(pdf_exp) * coef
        pdf = np.prod(pdf, axis=2) * np.power(10, 4 + self.slider_stats['threshold'] / 5)
        pdf = np.array(np.minimum(pdf, 255), dtype=np.uint8)

        blurred = cv2.GaussianBlur(pdf, make_kernel(self.slider_stats['blur'], False), 0)
        _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, make_kernel(self.slider_stats['open']))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, make_kernel(self.slider_stats['close']))

        return closed


class ComponentSample(DataSample):
    def __init__(self, input_data=None, component_name=None):
        super_data = None
        if input_data is not None:
            super_data = input_data['stuff']
        super().__init__(input_data)
        self.component_name = component_name
        self.color = ColorSample()
        self.contour = None

    def get_contours(self, binarized):
        contours, h = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if self.contour is not None:
            new_contours = []
            for contour in contours:
                match = cv2.matchShapes(self.contour, contour, cv2.CONTOURS_MATCH_I3, 0)
                print(f"Match: {match}")
                if match < self.color.slider_stats['contour threshold'] / 100:
                    new_contours.append(contour)
            contours = new_contours
        return contours

    def define_contour(self, image, x, y):
        color_binary = self.color.binarize_image(image)
        if color_binary is None:
            return None
        contours = self.get_contours(color_binary)

        for contour in contours:
            dist = cv2.pointPolygonTest(contour, (x, y), False)
            if dist >= 0:  # Point clicked is in contour
                self.contour = contour
                return
        # No match found
        self.contour = None

    def process_image(self, image):
        color_binary = self.color.binarize_image(image)
        if color_binary is None:
            return image
        contours = self.get_contours(color_binary)
        bgr_binary = cv2.cvtColor(color_binary, cv2.COLOR_GRAY2BGR)
        with_contours = cv2.drawContours(bgr_binary, contours, -1, (255, 0, 0), 3)
        return with_contours

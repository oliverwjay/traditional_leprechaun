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

    # def __getstate__(self):
    #     """
    #     Builds a representation of the sample to pickle
    #     :return: Representation of the sample
    #     """
    #     state = {'data': self.data}
    #     return state


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
        self.slider_stats = {'open': 0, 'close': 0, 'blur': 0, 'threshold': 50}
        self.contour = None
        self.flag_save_contour = False

    def save_contour(self):
        self.flag_save_contour = True

    def process_image(self, image):
        if len(self.data) < 10:
            return image[:, :, 0]
        coef = 1/(2*np.pi*np.square(self.sd))
        diff_x_mu = image - self.mean
        pdf_exp = -np.square(diff_x_mu/self.sd)/2
        pdf = np.exp(pdf_exp)*coef
        pdf = np.prod(pdf, axis=2)*np.power(10, 4 + self.slider_stats['threshold']/5)
        pdf = np.array(np.minimum(pdf, 255), dtype=np.uint8)

        blurred = cv2.GaussianBlur(pdf, make_kernel(self.slider_stats['blur'], False), 0)
        _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, make_kernel(self.slider_stats['open']))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, make_kernel(self.slider_stats['close']))

        contours, h = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        w_contours = cv2.drawContours(closed, contours, 0, 127, 3)

        if self.flag_save_contour:
            self.contour = w_contours[0]
            self.flag_save_contour = False

        return w_contours


class ComponentSample(DataSample):
    def __init__(self, input_data=None, component_name=None):
        super_data = None
        if input_data is not None:
            super_data = input_data['stuff']
        super().__init__(input_data)
        self.component_name = component_name
        self.color = ColorSample()

    # def __getstate__(self):
    #     """
    #     Builds a representation of the sample to pickle
    #     :return: Representation of the sample
    #     """
    #     state = super().__getstate__()
    #     state['obj_name'] = self.component_name
    #     state['color'] = self.color
    #     return state

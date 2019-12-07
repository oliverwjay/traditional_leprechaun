import numpy as np
from scipy.stats import multivariate_normal
import os
import pickle


class DataSample:
    def __init__(self, input_data=None):
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
        if self.data is None:
            return
        data_array = np.array(self.data)
        self.mean = np.mean(data_array, axis=0)
        self.sd = np.std(data_array, axis=0)

    def add_data(self, new_data_point):
        self.data.append(new_data_point)

    def to_export(self):
        if self.data_file is not None:
            pickle.dump(self.data, open(self.data_file, "wb"))


class PixelSample(DataSample):

    def process_image(self, image):
        if len(self.data) < 10:
            return image[:, :, 0]
        coef = 1/(2*np.pi*np.square(self.sd))
        diff_x_mu = image - self.mean
        pdf_exp = -np.square(diff_x_mu/self.sd)/2
        pdf = np.exp(pdf_exp)*coef
        pdf = np.prod(pdf, axis=2)*np.power(10, 12)
        pdf = np.array(np.minimum(pdf, 255), dtype=np.uint8)
        return pdf


class ObjectSample(DataSample):
    def __init__(self, input_data=None, object_name=None):
        super_data = None
        if input_data is not None:
            super_data = input_data['stuff']
        super().__init__(input_data)
        self.object_name = object_name

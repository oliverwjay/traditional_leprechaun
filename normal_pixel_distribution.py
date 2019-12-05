import numpy as np
from scipy.stats import multivariate_normal


class ObjectSample:
    def __init__(self):
        self.data = []
        self.sd = None
        self.mean = None

    def calculate_stats(self):
        if self.data is None:
            return
        data_array = np.array(self.data)
        self.mean = np.mean(data_array, axis=0)
        self.sd = np.std(data_array, axis=0)

    def add_data(self, new_pixel):
        self.data.append(new_pixel)

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

    def process_pixel(self, pix):
        if len(self.data) < 10:
            return pix
        coef = 1/(2*np.pi*np.square(self.sd))
        diff_x_mu = pix - self.mean
        pdf_exp = -np.square(diff_x_mu/self.sd)/2
        pdf = np.exp(pdf_exp)*coef
        pdf = np.prod(pdf)
        return pdf**np.power(10, 12)

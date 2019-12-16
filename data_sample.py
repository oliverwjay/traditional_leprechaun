import numpy as np
import cv2
import os
import pickle


def angle_wrap(a1, full_wrap=180):
    """
    Finds smallest representation of an angle. Useful for HSV distance
    :param a1: angle
    :param full_wrap: Max size
    :return: Smallest value
    """
    return (a1 - full_wrap / 2) % full_wrap - full_wrap / 2


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
        """
        Exports data to pickle file
        :return: None
        """
        if self.data_file is not None:
            pickle.dump(self.data, open(self.data_file, "wb"))


def make_kernel(k_size, kernel=True):
    """
    Builds a kernel for morphology
    :param k_size: Size of kernel
    :param kernel: True for full kernel matrix, False for tuple of size
    :return: Kernel
    """
    k_size |= 1
    k_dims = (k_size, k_size)
    if kernel:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, k_dims)
    else:
        return k_dims


def find_centroid(contour):
    """
    Finds the centroid of a contour
    :param contour: Contour to find the centroid of
    :return: Center point
    """
    m = cv2.moments(contour)
    if m['m00'] == 0:
        return 0, 0
    return int(m['m10']/m['m00']), int(m['m01']/m['m00'])


def find_center(p1, p2):
    """
    Helper function to find the center between two points
    :param p1: First point
    :param p2: Second point
    :return: Mid point
    """
    x = int((p1[0] + p2[0])/2)
    y = int((p1[1] + p2[1])/2)
    return x, y


class ColorSample(DataSample):
    """
    Stores data for a sample from a color
    """
    def __init__(self, input_data=None):
        """
        Constructor for ColorSample
        :param input_data: Data to build with pickle
        """
        super().__init__(input_data)
        self.slider_stats = {'open': 0, 'close': 0, 'blur': 0, 'threshold': 50, 'contour threshold': 50}
        self.contour = None
        self.save_steps = False

    def binarize_image(self, image):
        """
        Binarizes the image by how well it matches the gaussian model
        :param image: HSV image to process
        :return: Binary grayscale image
        """
        # Skip processing if there is not enough data
        if len(self.data) < 10:
            return None

        # Calculate probability density function
        coef = 1 / (2 * np.pi * np.square(self.sd))
        diff_x_mu = image - self.mean
        diff_x_mu[:, :, 0] = np.abs(angle_wrap(diff_x_mu[:, :, 0]))
        pdf_exp = -np.square(diff_x_mu / self.sd) / 2
        pdf = np.exp(pdf_exp) * coef
        pdf = np.prod(pdf, axis=2) * np.power(10, 4 + self.slider_stats['threshold'] / 5)
        pdf = np.array(np.minimum(pdf, 255), dtype=np.uint8)

        # For debugging / documentation
        if self.save_steps:
            cv2.imwrite("prob.jpg", pdf)

        # Blur image
        blurred = cv2.GaussianBlur(pdf, make_kernel(self.slider_stats['blur'], False), 0)

        # Binarize
        _, thresholded = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)

        # Save blurred image
        if self.save_steps:
            cv2.imwrite("blur_thresh.jpg", thresholded)

        # Open and close image
        opened = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, make_kernel(self.slider_stats['open']))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, make_kernel(self.slider_stats['close']))

        if self.save_steps:
            cv2.imwrite("morphed.jpg", closed)

        # Return processed image
        return closed


class ComponentSample(DataSample):
    """
    Handles data about a component from an object (eg beard, shirt)
    """
    def __init__(self, input_data=None, component_name=None):
        """
        Constructor for component sample
        :param input_data: To initialize with pickle
        :param component_name: Name of the component
        """
        super().__init__(input_data)
        self.component_name = component_name
        self.color = ColorSample()
        self.contour = None
        self.found_contours = []
        self.expected_size = None
        self.exp_poses = []

    def get_contours(self, binarized):
        """
        Find all matching contours
        :param binarized: Binarized image to find contours in
        :return: List of matching contours
        """
        # Find all contours
        contours, h = cv2.findContours(binarized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Save image with progress
        if self.color.save_steps:
            bgr_binary = cv2.cvtColor(binarized, cv2.COLOR_GRAY2BGR)
            with_contours = cv2.drawContours(bgr_binary, contours, -1, (255, 0, 0), 3)
            cv2.imwrite("all_contours.jpg", with_contours)

        # Filter out small contours
        contours = [contour for contour in contours if cv2.contourArea(contour) > 300]

        # If there is a model, filter out contours that don't match
        if self.contour is not None:
            new_contours = []
            for contour in contours:
                # Check if each contour matches
                match = cv2.matchShapes(self.contour, contour, cv2.CONTOURS_MATCH_I3, 0)
                if match < self.color.slider_stats['contour threshold'] / 100:
                    new_contours.append(contour)
            contours = new_contours

        # Return matching contours
        return contours

    def define_contour(self, image, x, y):
        """
        Sets the contour enclosing the clicked pixel as the model contour
        :param image: Image with contour
        :param x: Clicked x coordinate
        :param y: Clicked y coordinate
        :return: Contour matched
        """
        # Binarize image from color model
        color_binary = self.color.binarize_image(image)
        if color_binary is None:  # No color model
            return None

        # Get contours
        contours = self.get_contours(color_binary)

        # Find enclosing contours
        for contour in contours:
            dist = cv2.pointPolygonTest(contour, (x, y), False)
            if dist >= 0:  # Point clicked is in contour
                # Set as contour and return it
                self.contour = contour
                return contour
        # No match found
        self.contour = None

    def process_image(self, image):
        """
        Finds and overlays contours for an input image
        :param image: HSV image to analyze
        :return: Image with overlay
        """
        # Clear list of matching contours
        self.found_contours = []

        # Binarize color
        color_binary = self.color.binarize_image(image)
        if color_binary is None:  # No color model
            return np.zeros(image.shape, dtype=np.uint8)

        # Find contours
        contours = self.get_contours(color_binary)

        # BGR color image from grayscale for overlay
        bgr_binary = cv2.cvtColor(color_binary, cv2.COLOR_GRAY2BGR)

        # Output images for debugging
        if self.color.save_steps:
            with_contours = cv2.drawContours(bgr_binary.copy(), contours, -1, (255, 0, 0), 3)
            cv2.imwrite("filtered_contours.jpg", with_contours)

        # Check each contour for defects
        for contour in contours:
            # Find convex hull
            hull = cv2.convexHull(contour, returnPoints=False)
            hull_points = cv2.convexHull(contour, returnPoints=True)

            # Find convexity defects
            defects = cv2.convexityDefects(contour, hull)
            bgr_binary = cv2.drawContours(bgr_binary, [hull_points], -1, (255, 0, 0), 3)

            # Save progress
            if self.color.save_steps:
                cv2.imwrite("hull.jpg", bgr_binary)

            # Find object centroid
            centroid = find_centroid(contour)

            # Show on overlay
            cv2.circle(bgr_binary, centroid, 5, [255, 0, 0], -1)

            # Process defects
            if defects is not None:
                # Find biggest defect
                d, s, e = max((d, s, e) for s, e, f, d in defects[:, 0])
                start = tuple(contour[s][0])
                end = tuple(contour[e][0])
                gap_center = find_center(start, end)
                cv2.line(bgr_binary, centroid, gap_center, [0, 255, 0], 2)

                if self.color.save_steps:
                    cv2.line(bgr_binary, start, end, [0, 0, 255], 2)
                    cv2.imwrite("with_defect.jpg", bgr_binary)

                # Find orientation
                orientation = np.arctan2(centroid[0] - gap_center[0], centroid[1] - gap_center[1])

                # Get radius
                _, radius = cv2.minEnclosingCircle(contour)
                centroid = np.array(centroid)

                # Save matching contours
                self.found_contours.append({'orientation': orientation, 'centroid': centroid,
                                            'size': radius, 'contour': contour})
        # Return overlay
        return bgr_binary

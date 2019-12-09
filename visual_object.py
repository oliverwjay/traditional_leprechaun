import pickle
from os import path
import numpy as np
import cv2
from data_sample import ComponentSample, ColorSample


class VisualObject:
    def __init__(self, data_file=None, component_names=None):
        """
        Initializes the VisualObject from a file. Updates the keys to match given component names
        :param data_file: File name to read and save model data
        :param component_names:
        """
        self.data_file = data_file  # Save file name to write changes
        self.save_size_flag = False
        self.obj_unit_vector = None
        self.obj_dist = None
        self.obj_orientation = None
        self.obj_vect = None
        self.origin = None

        # Read model from file
        if data_file is not None and path.isfile(data_file):
            self.components = pickle.load(open(data_file, "rb"))
        else:  # Create empty model if needed
            raw_component_data = dict()

            # Update raw data to match given names
            if component_names is not None:
                # Add missing components
                for component in component_names:
                    if component not in raw_component_data.keys():
                        raw_component_data[component] = None

                # Remove unneeded component names
                for component in raw_component_data.keys():
                    if component not in component_names:
                        del raw_component_data[component]

            # Build class structure from raw data
            self.components = dict()  # Dictionary to contain component samples
            for component, data in raw_component_data.items():  # Populate
                self.components[component] = ComponentSample(data, component)  # Create object from data

    def add_contour(self, x, y):
        self.save_size_flag = (x, y)

    def get_contour_pose(self, contour):
        rel_contour_size = contour['size'] / self.obj_dist
        scaled_contour_position = (contour['centroid'] - self.origin) / self.obj_dist
        return np.concatenate((self.obj_unit_vector * scaled_contour_position, [rel_contour_size]))

    def clear_component(self, component_name):
        """
        Clears color for a given component
        :param component_name: Name of component to reset
        :return: None
        """
        self.components[component_name].color = ColorSample(None)

    def save(self):
        """
        Stores to pickle file
        :return: None
        """
        pickle.dump(self.components, open(self.data_file, "wb"))

    def match_components(self, img):
        """
        Draws contours that match the current object pose
        :return: Image with contours drawn on
        """
        for component in self.components.values():
            for contour in component.found_contours:
                invariant_pose = self.get_contour_pose(contour)
                if self.save_size_flag:
                    dist = cv2.pointPolygonTest(contour['contour'], self.save_size_flag, False)
                    if dist >= 0:  # Point clicked is in contour
                        component.exp_poses.append(invariant_pose)
                        self.save_size_flag = None
                for exp_pose in component.exp_poses:
                    if max(np.abs(exp_pose - invariant_pose)) < .2:
                        # Match fit
                        output = cv2.drawContours(img, [contour['contour']], -1, (0, 0, 255), 3)
        return img


class Leprechaun (VisualObject):
    def __init__(self):
        super().__init__("leprechaun.npy", ["Beard", "Hat", "Shirt", "Clover", "Skin"])

    def find_leprechaun(self, img):
        output = img.copy()
        shirts = self.components["Shirt"].found_contours
        beards = self.components["Beard"].found_contours

        for shirt in shirts:
            for beard in beards:
                self.obj_vect = shirt['centroid'] - beard['centroid']
                self.obj_dist = np.linalg.norm(self.obj_vect)
                self.obj_orientation = np.arctan2(self.obj_vect[0], self.obj_vect[1])
                self.obj_unit_vector = self.obj_vect / self.obj_dist
                self.origin = shirt["centroid"]

                output = self.match_components(output)
        return output


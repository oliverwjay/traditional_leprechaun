import pickle
from os import path
import numpy as np
import cv2
from data_sample import ComponentSample


class VisualObject:
    def __init__(self, data_file=None, component_names=None):
        """
        Initializes the VisualObject from a file. Updates the keys to match given component names
        :param data_file: File name to read and save model data
        :param component_names:
        """
        self.data_file = data_file  # Save file name to write changes

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

    def save(self):
        """
        Stores to pickle file
        :return: None
        """
        pickle.dump(self.components, open(self.data_file, "wb"))


class Leprechaun (VisualObject):
    def __init__(self):
        super().__init__("leprechaun.npy", ["Beard", "Hat", "Shirt", "Clover", "Face", "Hands"])
        self.save_size_flag = False

    def save_size(self):
        self.save_size_flag = True

    def find_leprechaun(self, img):
        shirts = self.components["Shirt"].found_contours
        beards = self.components["Beard"].found_contours

        for shirt in shirts:
            for beard in beards:
                obj_vect = shirt['centroid'] - beard['centroid']
                obj_dist = np.linalg.norm(obj_vect)
                obj_orientation = np.arctan2(obj_vect[0], obj_vect[1])
                rel_beard_size = beard['size'] / obj_dist
                rel_beard_orientation = beard['orientation'] - obj_orientation
                rel_shirt_size = shirt['size'] / obj_dist

                if self.save_size_flag:
                    self.save_size_flag = False
                    self.components["Shirt"].expected_size = rel_shirt_size
                    self.components["Beard"].expected_size = rel_beard_size

                exp_shirt_size = self.components["Shirt"].expected_size
                exp_beard_size = self.components["Beard"].expected_size
                if exp_shirt_size is not None and exp_beard_size is not None:
                    beard_size_error = np.abs(rel_beard_size - exp_beard_size)
                    rel_beard_orientation = ((rel_beard_orientation + np.pi) % (np.pi * 2)) - np.pi
                    beard_orientation_error = np.abs(rel_beard_orientation/(2 * np.pi))
                    shirt_size_error = np.abs(rel_shirt_size - exp_shirt_size)
                    print(beard_size_error, beard_orientation_error, shirt_size_error)
                    if max(beard_orientation_error, beard_size_error, shirt_size_error) < .15:
                        print("Leprechaun detected!")
                        img = cv2.line(img, tuple(beard['centroid']), tuple(shirt['centroid']), [0, 255, 0], 2)
                        img = cv2.drawContours(img, [shirt['contour']], -1, (0, 0, 255), 3)
                        img = cv2.drawContours(img, [beard['contour']], -1, (255, 0, 0), 3)
        return img


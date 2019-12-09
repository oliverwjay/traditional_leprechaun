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

    def clear_component(self, component_name):
        """
        Clears all data for a given component
        :param component_name: Name of component to reset
        :return: None
        """
        self.components[component_name] = ComponentSample(None, component_name)

    def save(self):
        """
        Stores to pickle file
        :return: None
        """
        pickle.dump(self.components, open(self.data_file, "wb"))


class Leprechaun (VisualObject):
    def __init__(self):
        super().__init__("leprechaun.npy", ["Beard", "Hat", "Shirt", "Clover", "Skin"])
        self.save_size_flag = False

    def save_size(self):
        self.save_size_flag = True

    def find_leprechaun(self, img):
        output = img.copy()
        shirts = self.components["Shirt"].found_contours
        beards = self.components["Beard"].found_contours

        for shirt in shirts:
            for beard in beards:
                obj_vect = shirt['centroid'] - beard['centroid']
                obj_dist = np.linalg.norm(obj_vect)
                obj_orientation = np.arctan2(obj_vect[0], obj_vect[1])
                obj_unit_vector = obj_vect / obj_dist

                for component in self.components.values():
                    if self.save_size_flag:
                        component.exp_poses = []
                    for contour in component.found_contours:
                        rel_contour_size = contour['size'] / obj_dist
                        scaled_contour_position = (contour['centroid'] - shirt['centroid']) / obj_dist
                        invariant_pose = np.concatenate((obj_unit_vector * scaled_contour_position, [rel_contour_size]))
                        if self.save_size_flag:
                            component.exp_poses.append(invariant_pose)
                        for exp_pose in component.exp_poses:
                            if max(np.abs(exp_pose - invariant_pose)) < .2:
                                # Match fit
                                output = cv2.drawContours(output, [contour['contour']], -1, (0, 0, 255), 3)
                            else:
                                print("Failed ", component.component_name)
        self.save_size_flag = False
        return output


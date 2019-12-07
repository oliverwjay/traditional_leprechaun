import pickle
from os import path
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
            raw_component_data = pickle.load(open(data_file, "rb"))
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

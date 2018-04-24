########################################################################
#
# Class for creating a data-set consisting of all files in a directory.
#
########################################################################

import numpy as np
import os
from cache import cache

########################################################################


def one_hot_encoded(class_numbers, num_classes=None):


    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is zero.
    if num_classes is None:
        num_classes = np.max(class_numbers) + 1

    return np.eye(num_classes, dtype=float)[class_numbers]


########################################################################


class DataSet:
    def __init__(self, in_dir, exts='.jpg'):


        # Extend the input directory to the full path.
        in_dir = os.path.abspath(in_dir)

        # Input directory.
        self.in_dir = in_dir

        # Convert all file-extensions to lower-case.
        self.exts = tuple(ext.lower() for ext in exts)

        # Names for the classes.
        self.class_names = []

        # Filenames for all the files in the training-set.
        self.filenames = []

        # Filenames for all the files in the test-set.
        self.filenames_test = []

        # Class-number for each file in the training-set.
        self.class_numbers = []

        # Class-number for each file in the test-set.
        self.class_numbers_test = []

        # Total number of classes in the data-set.
        self.num_classes = 0

        # For all files/dirs in the input directory.
        for name in os.listdir(in_dir):
            # Full path for the file / dir.
            current_dir = os.path.join(in_dir, name)

            # If it is a directory.
            if os.path.isdir(current_dir):
                # Add the dir-name to the list of class-names.
                self.class_names.append(name)

                # Training-set.

                # Get all the valid filenames in the dir (not sub-dirs).
                filenames = self._get_filenames(current_dir)

                # Append them to the list of all filenames for the training-set.
                self.filenames.extend(filenames)

                # The class-number for this class.
                class_number = self.num_classes

                # Create an array of class-numbers.
                class_numbers = [class_number] * len(filenames)

                # Append them to the list of all class-numbers for the training-set.
                self.class_numbers.extend(class_numbers)

                # Test-set.

                # Get all the valid filenames in the sub-dir named 'test'.
                filenames_test = self._get_filenames(os.path.join(current_dir, 'test'))

                # Append them to the list of all filenames for the test-set.
                self.filenames_test.extend(filenames_test)

                # Create an array of class-numbers.
                class_numbers = [class_number] * len(filenames_test)

                # Append them to the list of all class-numbers for the test-set.
                self.class_numbers_test.extend(class_numbers)

                # Increase the total number of classes in the data-set.
                self.num_classes += 1

    def _get_filenames(self, dir):

        # Initialize empty list.
        filenames = []

        # If the directory exists.
        if os.path.exists(dir):
            # Get all the filenames with matching extensions.
            for filename in os.listdir(dir):
                if filename.lower().endswith(self.exts):
                    filenames.append(filename)

        return filenames

    def get_paths(self, test=False):


        if test:
            # Use the filenames and class-numbers for the test-set.
            filenames = self.filenames_test
            class_numbers = self.class_numbers_test

            # Sub-dir for test-set.
            test_dir = "test/"
        else:
            # Use the filenames and class-numbers for the training-set.
            filenames = self.filenames
            class_numbers = self.class_numbers

            # Don't use a sub-dir for test-set.
            test_dir = ""

        for filename, cls in zip(filenames, class_numbers):
            # Full path-name for the file.
            path = os.path.join(self.in_dir, self.class_names[cls], test_dir, filename)

            yield path

    def get_training_set(self):


        return list(self.get_paths()), \
               np.asarray(self.class_numbers), \
               one_hot_encoded(class_numbers=self.class_numbers,
                               num_classes=self.num_classes)

    def get_test_set(self):


        return list(self.get_paths(test=True)), \
               np.asarray(self.class_numbers_test), \
               one_hot_encoded(class_numbers=self.class_numbers_test,
                               num_classes=self.num_classes)


########################################################################


def load_cached(cache_path, in_dir):


    print("Creating dataset from the files in: " + in_dir)

    # If the object-instance for DataSet(in_dir=data_dir) already
    # exists in the cache-file then reload it, otherwise create
    # an object instance and save it to the cache-file for next time.
    dataset = cache(cache_path=cache_path,
                    fn=DataSet, in_dir=in_dir)

    return dataset


########################################################################

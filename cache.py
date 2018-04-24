########################################################################
#
# Cache-wrapper for a function or class.

########################################################################

import os
import pickle
import numpy as np

########################################################################


def cache(cache_path, fn, *args, **kwargs):

    # If the cache-file exists.
    if os.path.exists(cache_path):
        # Load the cached data from the file.
        with open(cache_path, mode='rb') as file:
            obj = pickle.load(file)

        print("- Data loaded from cache-file: " + cache_path)
    else:
        # The cache-file does not exist.

        # Call the function / class-init with the supplied arguments.
        obj = fn(*args, **kwargs)

        # Save the data to a cache-file.
        with open(cache_path, mode='wb') as file:
            pickle.dump(obj, file)

        print("- Data saved to cache-file: " + cache_path)

    return obj


########################################################################


def convert_numpy2pickle(in_path, out_path):

    # Load the data using numpy.
    data = np.load(in_path)

    # Save the data using pickle.
    with open(out_path, mode='wb') as file:
        pickle.dump(data, file)


########################################################################

if __name__ == '__main__':

    def expensive_function(a, b):
        return a * b

    print('Computing expensive_function() ...')

    result = cache(cache_path='cache_expensive_function.pkl',
                   fn=expensive_function, a=123, b=456)

    print('result =', result)

    # Newline.
    print()

    class ExpensiveClass:
        def __init__(self, c, d):
            self.c = c
            self.d = d
            self.result = c * d

        def print_result(self):
            print('c =', self.c)
            print('d =', self.d)
            print('result = c * d =', self.result)

    print('Creating object from ExpensiveClass() ...')

    obj = cache(cache_path='cache_ExpensiveClass.pkl',
                fn=ExpensiveClass, c=123, d=456)

    obj.print_result()

########################################################################

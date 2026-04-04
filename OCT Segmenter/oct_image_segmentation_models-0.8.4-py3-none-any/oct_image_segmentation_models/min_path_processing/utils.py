import numpy as np


def generate_boundary(img_array, axis=0):
    """
    Convention used by code in model: Considering the image in a
    top to bottom fashion, the areas do not include the pixel of the
    boundary of which it ends; in other words, boundaries belong to
    the first pixel of the "next region". For example: If the first
    boundary is an array of 2's the top region will be height 2 (0th
    and 1st row)
    """
    boundaries = []
    num_classes = np.amax(img_array)

    for i in range(1, num_classes + 1):
        boundaries.append([x for x in np.argmax(img_array == i, axis=axis)])
    return np.array(boundaries)

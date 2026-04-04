import numpy as np
from skimage.util import random_noise
import time


def augment_dataset(images, masks, segs, aug_fn_arg):
    start_augment_time = time.time()

    num_images = len(images)
    aug_fn = aug_fn_arg[0]
    aug_arg = aug_fn_arg[1]

    augmented_images = np.zeros_like(images)
    augmented_masks = np.zeros_like(masks)
    augmented_segs = np.zeros_like(segs)

    for i in range(num_images):
        image = images[i, :, :]
        mask = masks[i, :, :]
        seg = segs[i, :, :]
        (
            augmented_images[i, :, :],
            augmented_masks[i, :, :],
            augmented_segs[i, :, :],
            _,
            _,
        ) = aug_fn(image, mask, seg, aug_arg)

    aug_desc = aug_fn(None, None, aug_arg, True)

    end_augment_time = time.time()
    total_aug_time = end_augment_time - start_augment_time

    return [
        augmented_images,
        augmented_masks,
        augmented_segs,
        aug_desc,
        total_aug_time,
    ]


def no_aug(image, mask, _aug_args, desc_only=False):
    if desc_only is False:
        return image, mask
    else:
        desc = "no aug"
        return desc


def flip_aug(image, mask, aug_args, desc_only=False):
    flip_type = aug_args["flip_type"]

    if flip_type == "up-down":
        axis = 0
    elif flip_type == "left-right":
        axis = 1

    if desc_only is False:
        aug_image = np.flip(image, axis=axis)
        if mask is not None:
            aug_mask = np.flip(mask, axis=axis)
        else:
            aug_mask = None

        return aug_image, aug_mask
    else:
        aug_desc = "flip aug: " + flip_type
        return aug_desc


def add_noise_aug(image, mask, aug_args, desc_only=False):
    if desc_only is False:
        mode = aug_args["mode"]
        mean = aug_args["mean"]
        variance = aug_args["variance"]

        """
        Documentation note on random_noise():
        Because of the prevalence of exclusively positive floating-point
        images in intermediate calculations, it is not possible to intuit
        if an input is signed based on dtype alone. Instead, negative values
        are explicitly searched for. Only if found does this function assume
        signed input.

        Returns:
        out: ndarray: Output floating-point image data on range [0, 1] or
        [-1, 1] if the input image was unsigned or signed, respectively.

        This is important because some models scale the input from [0, 1]
        (i.e. U-Net) while others [-1, 1] (i.e. keras.applications.ResNet50)
        """
        noise_img = random_noise(image, mode=mode, mean=mean, var=variance)
        return noise_img, mask
    else:
        return "add noise: " + str(aug_args)


augmentation_map = {
    "add_noise": add_noise_aug,
    "flip": flip_aug,
    "no_augmentation": no_aug,
}


def normalize(x):
    x = np.asarray(x)
    return (x - x.min()) / (np.ptp(x))

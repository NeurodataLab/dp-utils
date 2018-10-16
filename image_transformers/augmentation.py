import imgaug.augmenters as iaa


def get_light_augmenter():
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.2), keep_size=True),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.1))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-15, 15)
        )
    ], random_order=True)
    return seq


def get_light_augmentation_func(for_list=False, deterministic=False):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5),  # horizontal flips
        iaa.Crop(percent=(0, 0.2), keep_size=True),  # random crops
        # Small gaussian blur with random sigma between 0 and 0.5.
        # But we only blur about 50% of all images.
        iaa.Sometimes(0.5,
                      iaa.GaussianBlur(sigma=(0, 0.1))
                      ),
        # Strengthen or weaken the contrast in each image.
        iaa.Multiply((0.8, 1.2), per_channel=0.2),
        # Apply affine transformations to each image.
        # Scale/zoom them, translate/move them, rotate them and shear them.
        iaa.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-15, 15)
        )
    ], random_order=True)
    if deterministic:
        seq = seq.to_deterministic()
    if for_list:
        return seq.augment_images
    else:
        return seq.augment_image

def test_time_augmentation(image):
    aug1 = iaa.Pad(percent=(0., 0.2, 0.2, 0.), keep_size=True)
    aug2 = iaa.Pad(percent=(0.2, 0., 0.2, 0.), keep_size=True)
    aug3 = iaa.Pad(percent=(0.2, 0., 0., 0.2), keep_size=True)
    aug4 = iaa.Affine(rotate=15)
    aug5 = iaa.Fliplr()
    return [image] + [i.augment_image(image) for i in [aug1, aug2, aug3, aug4, aug5]]
import numpy as np
import cv2


def resize_frames(video_arr, target_size, keep_aspect_ratio=False):
    ret_list = []

    for im in video_arr:
        if keep_aspect_ratio:
            im_tmp = resize_image_keep_aspect(im, target_size=target_size)
            ret_list.append(cv2.resize(im_tmp, target_size))
        else:
            ret_list.append(cv2.resize(im, target_size))
    return np.array(ret_list)


def resize_image_keep_aspect(img, target_size=(500, 500)):
    """
    :param img:
    :param target_size: in h,w format
    """
    old_size = img.shape[:2]  # old_size is in (height, width) format

    ratios = [float(i) / float(j) for i, j in zip(target_size, old_size)]

    min_ratio_index = 0 if ratios[0] < ratios[1] else 1
    min_ratio = ratios[min_ratio_index]

    new_size = tuple([int(x * min_ratio) for x in old_size])

    # new_size should be in (width, height) format
    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = target_size[1] - new_size[1]
    delta_h = target_size[0] - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    new_im = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return new_im

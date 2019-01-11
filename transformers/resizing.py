import numpy as np
import cv2
import logging

from .. import ROOT_LOGGER_NAME, ROOT_LOGGER_LEVEL
logger = logging.getLogger('{}.{}'.format(ROOT_LOGGER_NAME, __name__))
logger.setLevel(ROOT_LOGGER_LEVEL)


def resize_frames(video_arr, target_size, keep_aspect_ratio=False):
    """
    :param video_arr: THWC format array or list
    :param target_size: h, w
    :param keep_aspect_ratio: True/False
    :return:
    """
    ret_list = []

    for im in video_arr:
        if keep_aspect_ratio:
            im_tmp = resize_image_keep_aspect(im, target_size=target_size)
            ret_list.append(im_tmp)
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


def loop_video_size_casting(num_frames_param, num_frames):
    base_range = np.arange(0, num_frames)
    num_ranges_whole = num_frames_param / num_frames
    linspace_res = num_frames_param % num_frames

    ret_linspace = np.concatenate(num_ranges_whole * [base_range] + base_range[:linspace_res])
    return ret_linspace


def back_and_fourth_video_size_casting(num_frames_param, num_frames):
    base_linspace = np.arange(0, num_frames)
    num_range_whole = num_frames_param / num_frames
    range_res = num_frames_param % num_frames

    list_to_cat = []
    for cat_counter in range(num_range_whole):
        direction = 1 if cat_counter % 2 == 0 else -1
        list_to_cat.append(base_linspace[::direction])

    direction = 1 if num_range_whole % 2 == 0 else -1
    list_to_cat.append(base_linspace[::direction][:range_res])

    return np.concatenate(list_to_cat)


def make_random_beginning_video_size_casting(step=2):
    def random_beginning_video_size_casting(num_frames_param, num_frames):    
        if num_frames <= step * num_frames_param: 
            return np.linspace(0, num_frames, num_frames_param, endpoint=False).astype(int)
        else:
            beginning = np.random.randint(0, num_frames - step * num_frames_param)
            return np.arange(beginning, beginning + step * num_frames_param, step, dtype=int)
    return random_beginning_video_size_casting

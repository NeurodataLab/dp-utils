import numpy as np

from kungfutils.transformers.iou import diag_iou, full_iou


def rel_boxes_resize_square(boxes, old_shape):
    h0, w0 = old_shape

    dw0, dh0 = max(h0, w0) - w0, max(w0, h0) - h0
    w1, h1 = w0 + dw0, h0 + dh0

    box_abs = boxes * np.tile(old_shape[::-1], 2)
    box_abs[:, 0::2] += dw0 / 2
    box_abs[:, 1::2] += dh0 / 2

    box_abs[:, 0::2] /= w1
    box_abs[:, 1::2] /= h1

    return box_abs


def random_crop_with_constraints(boxes, labels, size, min_scale=0.3, max_scale=1., max_aspect_ratio=2,
                                 constraints=(0.3, 0.9), max_trial=10, target_shape=None):
    """
    :param boxes: (N, 4), in relative coordinates (xyxy)
    :param labels (N, ?)
    :param size: image_size
    :param min_scale:
    :param max_scale:
    :param max_aspect_ratio:
    :param constraints: (min IOU with crop box, min IOU with original box)
    :param max_trial: num of trials
    :param target_shape: size of image to be resized to,
                         will convert resulting boxes wrt to resizing and keeping aspect ratio
    :return:
    """
    assert target_shape is None or target_shape[0] == target_shape[1], 'Implemented only for None and square final size'

    h, w = size
    zero_crop = np.array((0, 0, w, h))

    new_boxes = boxes * np.array([w, h, w, h])

    min_iou_crop = constraints[0]
    min_iou_orig = constraints[1]

    scales = np.random.uniform(min_scale, max_scale, size=max_trial)

    aspect_ratios_low = np.maximum(1 / max_aspect_ratio, scales * scales)
    aspect_ratios_high = np.minimum(max_aspect_ratio, 1 / (scales * scales))

    aspect_ratios = aspect_ratios_low + np.random.uniform(size=max_trial) * (aspect_ratios_high - aspect_ratios_low)

    crop_hs = (h * scales / np.sqrt(aspect_ratios))
    crop_ws = (w * scales * np.sqrt(aspect_ratios))

    crop_ts = (h - crop_hs) * np.random.uniform(size=crop_hs.shape)
    crop_ls = (w - crop_ws) * np.random.uniform(size=crop_ws.shape)

    crop_bs = crop_ts + crop_hs
    crop_rs = crop_ls + crop_ws

    crop_boxes = np.stack([crop_ls, crop_ts, crop_rs, crop_bs], axis=-1)

    orig_boxes_cropped = np.array(
        [np.clip(new_boxes, a_min=np.tile(crop_box[:2], 2), a_max=np.tile(crop_box[2:], 2)) for crop_box in crop_boxes]
    )  # (num_attempts, num_boxes, 4)

    # filter 2
    crop_box_iou = full_iou(new_boxes, crop_boxes)  # (box_id, crop_id), (num_boxes, num_attempts)

    crop_selector = np.any(crop_box_iou > min_iou_crop, axis=0)
    if crop_selector.sum() == 0:
        if target_shape is not None:
            boxes = rel_boxes_resize_square(boxes=boxes, old_shape=target_shape)
        return zero_crop, (boxes, labels)

    crop_boxes = crop_boxes[crop_selector]
    orig_boxes_cropped = orig_boxes_cropped[crop_selector]

    # filter 1
    box_orig_iou = np.array([diag_iou(new_boxes, orig_boxes_c) for orig_boxes_c in orig_boxes_cropped])
    # (attempt_id, cur_box_id), (num_attempts, num_boxes)

    box_selector = box_orig_iou > min_iou_orig

    crop_selector = np.any(box_selector, axis=1)
    if crop_selector.sum() == 0:
        if target_shape is not None:
            boxes = rel_boxes_resize_square(boxes=boxes, old_shape=target_shape)
        return zero_crop, (boxes, labels)

    crop_boxes = crop_boxes[crop_selector]
    orig_boxes_cropped = orig_boxes_cropped[crop_selector]
    box_selector = box_selector[crop_selector]

    # choosing best crop id
    best_crop_id = np.argmax(box_selector.sum(axis=1))

    crop = crop_boxes[best_crop_id]
    xy_arr = np.tile(crop[:2], 2)
    whwh_arr = np.tile(crop[2:] - crop[:2], 2)

    new_boxes = (orig_boxes_cropped[best_crop_id] - xy_arr) / whwh_arr
    labels[np.logical_not(box_selector[best_crop_id]), :] = -1
    if target_shape is not None:
        new_boxes = rel_boxes_resize_square(boxes=new_boxes, old_shape=target_shape)

    return crop, (new_boxes, labels)

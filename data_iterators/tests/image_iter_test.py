import json
import pickle
import os.path as osp
import numpy as np
import pandas as pd
import cv2

from data_iterators.base_iterator import BaseIterator
from data_iterators.balancers.ohc_balancer import OHCBalancer

from data_iterators.preprocessors.image_preprocessor import RGBImagesFromCallable
from data_iterators.preprocessors.base_preprocessor import IdentityPreprocessor
from image_transformers.resizing import resize_image_keep_aspect

images_format_string = '/home/mininlab/DATA/EM/images/{}/images/'
no_faces_format_string = '/home/mininlab/DATA/EM/images/{}/images_no_faces/'


def test_transform(target_shape):
    def wrapped(img):
        img = resize_image_keep_aspect(img, target_size=target_shape)
        return img

    return wrapped


def fragment_to_arr_factory(frag2frame, bbox_data, position_data):
    frag2frames = json.load(open(frag2frame))
    position_data = json.load(open(position_data))
    bboxes = pickle.load(open(bbox_data, 'rb'))

    position_selector = {'left': 0, 'right': -1, 'alone': 0}

    def wrapped(frag_id):
        show_id = frag_id.split('/')[0]
        position = position_data[show_id]
        im_files = [osp.join(images_format_string.format(show_id), i) for i in frag2frames[frag_id]]

        bbox_centers = []
        bbox_index = range(len(bboxes[frag_id]))

        for bb in bboxes[frag_id]:
            x1, y1, x2, y2 = bb
            bbox_centers.append(((x1 + x2) / 2., (y1 + y2) / 2.))  # x,y

        needed_bbox_id = sorted(zip(bbox_centers, bbox_index), key=lambda x: x[0][0])[position_selector[position]]
        needed_bbox = bboxes[frag_id][needed_bbox_id[1]]
        x1, y1, x2, y2 = needed_bbox

        im_arr = []
        for i in im_files:
            img = cv2.cvtColor(cv2.imread(i), cv2.COLOR_BGR2RGB)

            cropped = img[x1: x2, y1:y2]
            im_arr.append(cropped)
        im_arr = np.array(im_arr)

        return im_arr

    return wrapped, list(bboxes.keys())


if __name__ == '__main__':
    target_size = 200, 200

    getter, avail_frags = fragment_to_arr_factory(
        frag2frame='/home/mininlab/DATA/EM/em_meta/fragments2frames.json',
        bbox_data='/home/mininlab/DATA/EM/em_meta/body_mask_rcnn_em.pkl',
        position_data='/home/mininlab/DATA/EM/em_meta/side_info.json'
    )

    labels_data = pd.read_csv('/home/mininlab/DATA/EM/potion_newbins_neutral_v7_train_data.csv')

    perm = np.random.permutation(labels_data.shape[0])
    labels_data = labels_data.iloc[perm]  # shuffle

    dummies_cols = ['ohc1_8', 'ohc1_9_21', 'ohc1_10_11', 'ohc1_12', 'ohc1_100']

    balancer = OHCBalancer(data=labels_data[dummies_cols].values, raise_on_end=True)

    data_proc = {
        'data': RGBImagesFromCallable(name='data', func=getter, num_frames=60, seq_transformer=test_transform(target_size))
    }

    label_proc = {
        'label': IdentityPreprocessor(name='label', shape=(len(dummies_cols),))
    }

    iter_train = BaseIterator(
        balancer=balancer, data={'data': labels_data['path']},
        label={'label': labels_data[dummies_cols].values},
        data_preprocessors=data_proc,
        label_preprocessors=label_proc,
        batch_size=32
    )

    idx_next = iter_train.next()

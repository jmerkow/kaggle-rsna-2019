from __future__ import absolute_import, print_function

import os

import h5py
import numpy as np
import pandas as pd
import torch
from albumentations.core.composition import BaseCompose
from torchvision.datasets.vision import VisionDataset

from ..dicom import sitk_read_image


def h5_read_image(fn):
    with h5py.File(fn, 'r') as f:
        array = np.array(f['data']).astype('float32')
    return array


readers = {
    'dcm': sitk_read_image,
    'h5': h5_read_image,
}


class RSNA2019Dataset(VisionDataset):

    label_map = {
        'any': 'any',
        'edh': 'epidural',
        'iph': 'intraparenchymal',
        'ivh': 'intraventricular',
        'sah': 'subarachnoid',
        'sdh': 'subdural',
    }

    def __init__(self, root, csv_file, transform=None, target_transform=None, transforms=None,
                 convert_rgb=True, preprocessing=None, img_ids=None,
                 reader='h5',
                 class_order=('sdh', 'sah', 'ivh', 'iph', 'edh', 'any'),
                 **filter_params):
        super(RSNA2019Dataset, self).__init__(root, transforms, transform, target_transform)

        self.data = pd.read_csv(csv_file).set_index('ImageId')

        assert all(c in self.label_map for c in class_order), "bad class order"
        self.class_order = class_order

        img_ids = img_ids or self.data.index.tolist()
        img_ids = self.apply_filter(img_ids, **filter_params)
        self.ids = {i: imgid for i, imgid in enumerate(img_ids)}
        self.rev_ids = {v: k for k, v in self.ids.items()}
        self.transforms_are_albumentation = isinstance(self.transforms, BaseCompose)
        self.convert_rgb = convert_rgb
        self.preprocessing = preprocessing

        assert reader in readers, 'bad reader type'

        self.image_ext = reader
        self.read_image = readers[reader]

    def apply_filter(self, img_ids, **filter_params):
        # place holder for now
        return img_ids

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]
        image_row = self.data.loc[img_id].to_dict()
        path = image_row['filepath']
        path = os.path.splitext(path)[0] + '.' + self.image_ext

        img = self.read_image(os.path.join(self.root, path))
        try:
            target = [(image_row['label__' + self.label_map[c]]) for c in self.class_order]
        except KeyError:
            target = None

        output = dict(image=img)
        if self.transforms is not None:
            if self.transforms_are_albumentation:
                output = self.transforms(**output)
            else:
                raise NotImplementedError('Not implemented yet, must be albumentation based transform')

        if self.preprocessing:
            if target is not None:
                target = torch.tensor(target).float()
            output = self.preprocessing(**output)

        output['index'] = index
        output['image_id'] = img_id
        if target is not None:
            output['target'] = target

        return output

    def __len__(self):
        return len(self.ids)

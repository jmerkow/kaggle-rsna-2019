from __future__ import absolute_import, print_function

import json
import os
import random
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import torch
from albumentations.core.composition import BaseCompose
from torchvision.datasets.vision import VisionDataset

from kaggle_lib.dicom import sitk_read_image
from kaggle_lib.pytorch.utils import Timer


def h5_read_image(fn):
    with h5py.File(fn, 'r') as f:
        array = np.array(f['data']).astype('float32')
    return array


class NumpyMemmapReader(object):
    __info_filename = 'memmap_info.json'
    __memmap_dat_filename = "memmap.dat"

    def __init__(self, img_dir, dtype='float32', data=None):
        self.dtype = dtype
        self.data = data
        with open(os.path.join(img_dir, self.__info_filename), 'r') as f:
            config = json.load(f)
            self.image_id_to_index = config['image_id_to_index']
            self.shape = tuple(config['info'].pop('shape'))
            self._dtype = config['info'].pop('dtype')
            self.info = config['info']

        self.memmap_path = os.path.join(img_dir, self.__memmap_dat_filename)
        self.memmap = np.memmap(self.memmap_path, mode='r', shape=self.shape, dtype=self._dtype, **self.info)

    def __getitem__(self, item):
        return self(item)

    def __call__(self, image_id):
        index = self.image_id_to_index[image_id]
        image = self.memmap[index, ...]
        if self.dtype is not None:
            image = image.astype(self.dtype)
        return image


class RSNA2019Dataset(VisionDataset):
    label_map = {
        'any': 'any',
        'edh': 'epidural',
        'iph': 'intraparenchymal',
        'ivh': 'intraventricular',
        'sah': 'subarachnoid',
        'sdh': 'subdural',
    }

    default_extra_fields = [
        'median_ipp_z_diff_norm',
        'mean_ipp_z_diff_norm',
        'median_ipp_z_diff',
        'mean_ipp_z_diff',
        'ipp_z'
    ]

    def __init__(self, root, csv_file, transform=None, target_transform=None, transforms=None,
                 convert_rgb=True, preprocessing=None, img_ids=None,
                 reader='h5',
                 class_order=('sdh', 'sah', 'ivh', 'iph', 'edh', 'any'),
                 limit=None,
                 tta_transform=None,
                 extra_fields=None,
                 **filter_params):

        self.timers = defaultdict(Timer)
        self.tta_transform = tta_transform

        assert reader in ['h5', 'dcm', 'memmap'], 'bad reader type'

        super(RSNA2019Dataset, self).__init__(root, transforms, transform, target_transform)

        self.reader_type = reader
        if self.reader_type == 'memmap':
            self.reader = NumpyMemmapReader(self.root, dtype='float32')

        data = pd.read_csv(csv_file).set_index('ImageId')

        data['fullpath'] = self.root + "/" + data['filepath']
        extra_fields = extra_fields or []
        self.extra_fields = set(extra_fields + self.default_extra_fields)
        missing_fields = self.extra_fields.difference(data)
        if len(missing_fields):
            raise ValueError('no fields: {}'.format(', '.join(missing_fields)))

        assert all(c in self.label_map for c in class_order), "bad class order"
        self.class_order = class_order
        img_ids = img_ids or data.index.tolist()
        if limit:
            random.shuffle(img_ids)
            img_ids = img_ids[:limit]
        img_ids = self.apply_filter(img_ids, **filter_params)
        self.ids = {i: imgid for i, imgid in enumerate(img_ids)}

        self._num_images = len(self.ids)
        self.data = data.loc[list(self.ids.values())].T.to_dict()

        self.rev_ids = {v: k for k, v in self.ids.items()}

        self.transforms_are_albumentation = isinstance(self.transforms, BaseCompose)
        self.convert_rgb = convert_rgb
        self.preprocessing = preprocessing

    def apply_filter(self, img_ids, **filter_params):
        return img_ids

    def read_image(self, image_id):
        # TODO: Make this cleaner
        if self.reader_type == 'dcm':
            path = self.data[image_id]['fullpath']
            return sitk_read_image(path)

        elif self.reader_type == 'h5':
            path = self.data[image_id]['fullpath']
            path = os.path.splitext(path)[0] + '.h5'
            return h5_read_image(path)

        elif self.reader_type == 'memmap':
            assert self.reader is not None
            return self.reader[image_id]

    def __repr__(self):
        body = []
        body += ["Reader: {}".format(self.reader_type)]
        body += ['Extra Fields: {}'.format(','.join(self.extra_fields))]
        lines = [" " * self._repr_indent + line for line in body]
        lines.insert(0, super().__repr__())
        return '\n'.join(lines)

    def _get_extra_fields(self, image_row):
        output = {x: np.atleast_1d((pd.to_numeric(image_row[x], errors='coerce'))) for x in self.extra_fields}

        if self.preprocessing:
            output = {k: torch.tensor(v) for k, v in output.items()}

        return output

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        self.timers['getitem'].tic()
        img_id = self.ids[index]
        image_row = self.data[img_id]

        try:
            target = [(image_row['label__' + self.label_map[c]]) for c in self.class_order]
        except KeyError:
            target = None

        self.timers['read_image'].tic()
        img = self.read_image(img_id)
        self.timers['read_image'].toc()

        self.timers['augmentation'].tic()
        output = dict(image=img)
        if self.transforms is not None:
            if self.transforms_are_albumentation:
                output = self.transforms(**output)
            else:
                raise NotImplementedError('Not implemented yet, must be albumentation based transform')
        self.timers['augmentation'].toc()

        if self.tta_transform is not None:
            images = self.tta_transform(output['image'])
            images = [self.preprocessing(image=image)['image'] for image in images]
            tmp = torch.stack(images)
            output['image'] = tmp
            if target is not None:
                target = torch.tensor(target).float()
        else:
            self.timers['preprocessing'].tic()
            if self.preprocessing:
                if target is not None:
                    target = torch.tensor(target).float()
                output = self.preprocessing(**output)
            self.timers['preprocessing'].toc()

        output['index'] = index
        output['image_id'] = img_id
        if target is not None:
            output['target'] = target

        output.update(self._get_extra_fields(image_row))
        self.timers['getitem'].toc()
        return output

    def __len__(self):
        return self._num_images

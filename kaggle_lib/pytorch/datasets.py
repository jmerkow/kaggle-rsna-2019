from __future__ import absolute_import, print_function

import json
import os
import random
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
import six
import torch
import yaml
from albumentations.core.composition import BaseCompose
from torch.utils.data.sampler import Sampler
from torchvision.datasets.vision import VisionDataset

from kaggle_lib.dicom import sitk_read_image
from kaggle_lib.pytorch.utils import Timer


class SequenceSampler(Sampler):

    def __init__(self, dataset, shuffle=False, replacement=False):
        self.shuffle = shuffle
        self.replacement = replacement
        self.sequences = list(dataset.sequences.values())
        self.num_samples = len(self.sequences)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        n = self.num_samples
        if self.shuffle:
            if self.replacement:
                indices = torch.randint(high=n, size=(n,), dtype=torch.int64).tolist()
            else:
                indices = torch.randperm(n).tolist()
        else:
            indices = range(n)
        return (self.sequences[i] for i in indices)


class SequenceBatchSampler(Sampler):
    """
    Wraps SequentialSampler to yield a mini-batch of sequential
    indices. For Kaggle RSNA2019 competition.

    Args:
        sampler (Sampler): SequentialSampler.
    """

    def __init__(self, dataset, **kwargs):
        self.dataset = dataset
        self.sampler = SequenceSampler(self.dataset, **kwargs)

    def __iter__(self):
        for sequence in self.sampler:
            self.dataset.reset_sequence()
            yield sequence

    def __len__(self):
        return len(self.sampler)


def get_data_constants(data_root='/data', filename='datacatalog.yml'):
    with open(os.path.join(data_root, filename), 'r') as f:
        config = yaml.safe_load(f)

    return config['datacatalog'], config['dataset_map']


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

    default_extra_fields = ['ipp_z']

    # [
    #     'median_ipp_z_diff_norm',
    #     'mean_ipp_z_diff_norm',
    #     'median_ipp_z_diff',
    #     'mean_ipp_z_diff',
    #     'ipp_z'
    # ]

    def __init__(self, root, csv_file, transform=None, target_transform=None, transforms=None,
                 convert_rgb=True, preprocessing=None, img_ids=None,
                 reader='h5',
                 class_order=('sdh', 'sah', 'ivh', 'iph', 'edh', 'any'),
                 limit=None,
                 tta_transform=None,
                 extra_fields=None,
                 extra_datasets=None,
                 sequence_key='series_instance_uid',
                 sequence_order_key='ipp_z',
                 sequence_mode=True,
                 **filter_params):

        self.timers = defaultdict(Timer)
        self.tta_transform = tta_transform
        self.sequence_key = sequence_key
        self.sequence_order_key = sequence_order_key

        assert reader in ['h5', 'dcm', 'memmap'], 'bad reader type'

        super(RSNA2019Dataset, self).__init__(root, transforms, transform, target_transform)

        print("ROOT:", self.root)

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

        extra_data = []
        if extra_datasets is not None:
            if isinstance(extra_datasets, six.string_types):
                extra_datasets = [extra_datasets]

            for extra_dataset in extra_datasets:
                datacatalog, dataset_map = get_data_constants('/data')
                dataset_dict = datacatalog[extra_dataset]
                extra_dataset_imgdir = os.path.join('/data/', dataset_dict['img_dir'])
                extra_dataset_csv = os.path.join('/data/', dataset_dict['csv_file'])
                df = pd.read_csv(extra_dataset_csv).set_index('ImageId')
                df['fullpath'] = extra_dataset_imgdir + "/" + df['filepath']
                extra_data.append(df)

        img_ids = img_ids or data.index.tolist()
        if limit:
            random.shuffle(img_ids)
            img_ids = img_ids[:limit]
        img_ids = self.apply_filter(img_ids, **filter_params)
        data = data.loc[img_ids].sort_values([self.sequence_key, self.sequence_order_key])

        self.original_data_len = len(data)

        if extra_data:
            data = pd.concat([data] + extra_data, sort=True)

        img_ids = data.index.tolist()
        self.ids = {i: imgid for i, imgid in enumerate(img_ids)}

        self._num_images = len(self.ids)
        data = data.loc[list(self.ids.values())].copy()
        self.rev_ids = {v: k for k, v in self.ids.items()}

        self.data = data.join(pd.Series(self.rev_ids, name='index'))

        self.sequences = self.data.groupby(self.sequence_key)['index'].apply(list).to_dict()

        self.transforms_are_albumentation = isinstance(self.transforms, BaseCompose)
        self.convert_rgb = convert_rgb
        self.preprocessing = preprocessing

        self.replay_params = None
        self.sequence_mode = sequence_mode

    def apply_filter(self, img_ids, **filter_params):
        return img_ids

    def read_image(self, image_id):
        # TODO: Make this cleaner
        if self.reader_type == 'dcm':
            path = self.data.loc[image_id]['fullpath']
            return sitk_read_image(path)

        elif self.reader_type == 'h5':
            path = self.data.loc[image_id]['fullpath']
            path = os.path.splitext(path)[0] + '.h5'
            return h5_read_image(path)

        elif self.reader_type == 'memmap':
            assert self.reader is not None
            return self.reader[image_id]

    def __super_repr__(self):
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        # if hasattr(self, "transforms") and self.transforms is not None:
        # body += [str(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def __repr__(self):
        body = []
        body += ["Original Data Count: {}, Extra Data Count: {}".format(self.original_data_len,
                                                                        self._num_images - self.original_data_len)]
        body += ["Reader: {}".format(self.reader_type)]
        body += ['Extra Fields: {}'.format(','.join(self.extra_fields))]
        lines = [" " * self._repr_indent + line for line in body]
        lines.insert(0, self.__super_repr__())
        return '\n'.join(lines)

    def _get_extra_fields(self, image_row):
        output = {x: np.atleast_1d((pd.to_numeric(image_row[x], errors='coerce'))) for x in self.extra_fields}

        if self.preprocessing:
            output = {k: torch.tensor(v).float() for k, v in output.items()}

        return output

    def reset_sequence(self):
        self.replay_params = None

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        self.timers['getitem'].tic()
        img_id = self.ids[index]
        image_row = self.data.loc[img_id]

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

                if self.replay_params is None or not self.sequence_mode:
                    output = self.transforms(**output)
                    self.replay_params = output.pop('replay', None)
                else:
                    output = self.transforms.replay(self.replay_params, **output)
                    output.pop('replay', None)
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


def get_csv_file(dataset_dict, data_root):
    if dataset_dict:
        return os.path.join(data_root, dataset_dict['csv_file'])


def get_dataset(dataset_dict, data_root, **kwargs):
    if dataset_dict:
        root = os.path.join(data_root, dataset_dict['img_dir'])
        csv_file = os.path.join(data_root, dataset_dict['csv_file'])
        return RSNA2019Dataset(root=root, csv_file=csv_file, **kwargs)

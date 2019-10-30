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


class LinearSampleScheduler(object):

    def __init__(self, sampler, start_rate=None, end_rate=None, cycle_length=6, last_epoch=-1):
        self.sampler = sampler
        self.true_rate = self.sampler.true_rate
        self.start_rate = start_rate if start_rate is not None else self.true_rate
        self.end_rate = end_rate if end_rate is not None else self.true_rate
        self.cycle_length = cycle_length
        self.pos_freqs = np.linspace(self.start_rate, self.end_rate, self.cycle_length, endpoint=True)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        curr_epoch = self.last_epoch
        if self.cycle_length is not None:
            curr_epoch = curr_epoch % self.cycle_length

        curr_epoch = min(curr_epoch, len(self.pos_freqs) - 1)
        self.sampler.pos_freq = self.pos_freqs[curr_epoch]


class StepSampleScheduler(object):

    def __init__(self, sampler, milestones=None, start_rate=None, end_rate=None, cycle_length=None, last_epoch=-1):
        self.sampler = sampler
        self.true_rate = self.sampler.true_rate
        self.start_rate = start_rate if start_rate is not None else self.true_rate
        self.end_rate = end_rate if end_rate is not None else self.true_rate
        self.cycle_length = cycle_length
        self.last_epoch = last_epoch

        self.milestones = np.array(milestones or [])
        self.pos_freqs = np.linspace(self.start_rate, self.end_rate, len(self.milestones) + 1, endpoint=True)

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch

        curr_epoch = self.last_epoch
        if self.cycle_length is not None:
            curr_epoch = curr_epoch % self.cycle_length

        i = np.min(np.append(np.where(self.milestones > curr_epoch), np.inf))
        if i == np.inf:
            i = -1
        self.sampler.pos_freq = self.pos_freqs[int(i)]


class LabelSampler(Sampler):
    schedulers = {
        'linear': LinearSampleScheduler,
        'step': StepSampleScheduler,
    }

    def _get_strata_indexes(self):
        data = self.dataset.data
        if self.strata == 'image':
            pos_df_indexes = data.query("label__{}>0".format(self.label)).index.tolist()
            neg_df_indexes = data.query("not label__{}>0".format(self.label)).index.tolist()

        elif self.strata == 'seq':
            seq_key = self.dataset.sequence_key
            strat = data.groupby(seq_key)['label__{}'.format(self.label)].max().to_frame()
            sp = strat.query("label__{}>0".format(self.label)).index.tolist()

            pos_df_indexes = data[data[seq_key].isin(sp)].index.tolist()
            neg_df_indexes = data[~data[seq_key].isin(sp)].index.tolist()
        else:
            raise NotImplementedError('strata {} not implemented'.format(self.strata))

        pos_idxs = [self.dataset.rev_ids[i] for i in pos_df_indexes]
        neg_idxs = [self.dataset.rev_ids[i] for i in neg_df_indexes]

        return pos_idxs, neg_idxs

    def __init__(self, dataset, last_epoch=-1,
                 sample_mode='over',
                 strata='image', label='any', scheduler='linear', **scheduler_params):

        self.strata = strata
        self.label = label
        self.dataset = dataset

        assert sample_mode in ['over', 'under'], 'bad sample mode'
        self.sample_mode = sample_mode

        self.pos_idxs, self.neg_idxs = self._get_strata_indexes()

        self.n_positive_true = len(self.pos_idxs)
        self.n_negative_true = len(self.neg_idxs)

        self.true_rate = self.n_positive_true / (self.n_negative_true + self.n_positive_true)
        self.scheduler = self.schedulers[scheduler](self, last_epoch=last_epoch, **scheduler_params)

        self.step()
        iter(self)

    def __iter__(self):

        if self.sample_mode == 'over':
            if self.pos_freq > 0:
                self.n_positive = int(self.n_negative_true * (self.pos_freq) / (1 - self.pos_freq))
            else:
                self.n_positive = 0
            self.n_negative = self.n_negative_true

            stack = np.random.choice(self.pos_idxs, size=self.n_positive, replace=len(self.pos_idxs) < self.n_positive)
            if self.pos_freq < 1.0:
                stack = np.hstack((stack, self.neg_idxs))

        elif self.sample_mode == 'under':
            if self.pos_freq < 1:
                self.n_negative = int(self.n_positive_true * (1 - self.pos_freq) / self.pos_freq)
            else:
                self.n_negative = 0
            self.n_positive = self.n_positive_true
            stack = np.random.choice(self.neg_idxs, size=self.n_negative, replace=len(self.neg_idxs) < self.n_negative)
            if self.pos_freq > 0:
                stack = np.hstack((stack, self.pos_idxs))
        shuffled = np.random.permutation(stack)
        return iter(shuffled.tolist())

    def step(self, epoch=None):
        self.scheduler.step(epoch=epoch)

    def __len__(self):

        return self.n_positive + self.n_negative


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
                 sequence_mode=False,
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
        data = data.loc[img_ids].sort_values([self.sequence_key, self.sequence_order_key])

        self.original_data_len = len(data)

        if extra_data:
            data = pd.concat([data] + extra_data, sort=True)

        self.extra_data_count = self.original_data_len - len(data)

        data = self.apply_filter(data, **filter_params)

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

    def apply_filter(self, data, positive_series_only=False, neg_from_neg_series=False, **kwargs):
        if positive_series_only or neg_from_neg_series:
            total = len(data)
            seq_with_any = (data.groupby(self.sequence_key)['label__any'].max().to_frame()
                            .query('label__any>0').index.tolist())

            if positive_series_only:
                data = data[data[self.sequence_key].isin(seq_with_any)].copy()

            if neg_from_neg_series:
                data['seq_label_any'] = data[self.sequence_key].isin(seq_with_any)
                data = data.query('(seq_label_any>0 and label__any>0) or seq_label_any == 0')

            print('postive series filter: {} -> {}'.format(total, len(data)))
        return data

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
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def __repr__(self):
        body = []
        body += ["Original Data Count: {}, Amount Extra Data: {}, Final (Filtered) Data Count: {}".format(
            self.original_data_len,
            self.extra_data_count,
            self._num_images)]
        body += ["Reader: {}".format(self.reader_type)]
        body += ['Extra Fields: {}'.format(','.join(self.extra_fields))]

        if hasattr(self, "transforms") and self.transforms is not None:
            body += ['Transform:', '\n'.join([" " * self._repr_indent + s for s in str(self.transforms).split('\n')])]

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

    def transform_image(self, image):

        output = dict(image=image)
        if self.transforms is not None:
            if self.replay_params is None or not self.sequence_mode:
                output = self.transforms(**output)
                self.replay_params = output.pop('replay', None)
            else:
                output = self.transforms.replay(self.replay_params, **output)
                output.pop('replay', None)
        if self.preprocessing:
            output = self.preprocessing(**output)

        return output['image']

    def transform_target(self, target=None):
        if self.preprocessing:
            if target is not None:
                target = torch.tensor(target).float()
        return target

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
        output = dict()
        if self.tta_transform is not None:
            images = self.tta_transform(img)
            images = [self.transform_image(image) for image in images]
            output['image'] = torch.stack(images)
        else:
            output['image'] = self.transform_image(img)
        self.timers['augmentation'].toc()

        if target is not None:
            output['target'] = self.transform_target(target)

        output['index'] = index
        output['image_id'] = img_id

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

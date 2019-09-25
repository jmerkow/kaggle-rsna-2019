from __future__ import absolute_import, print_function

import os

import SimpleITK as sitk
import pandas as pd
import torch
from albumentations.core.composition import BaseCompose
from torchvision.datasets.vision import VisionDataset


class RSNA2019Dataset(VisionDataset):
    label_map = {
        'any': 'label__any',
        'edh': 'label__epidural',
        'iph': 'label__intraparenchymal',
        'ivh': 'label__intraventricular',
        'sah': 'label__subarachnoid',
        'sdh': 'label__subdural',

    }

    def __init__(self, root, csv_file, transform=None, target_transform=None, transforms=None,
                 convert_rgb=True, preprocessing=None, img_ids=None,
                 class_order=None, **filter_params):
        super(RSNA2019Dataset, self).__init__(root, transforms, transform, target_transform)

        self.data = pd.read_csv(csv_file).set_index('ImageId')

        if class_order is None:
            class_order = self.label_map.keys()

        assert all(c in self.label_map for c in class_order), "bad class order"
        self.class_order = class_order

        img_ids = img_ids or self.data.index.tolist()
        img_ids = self.apply_filter(img_ids, **filter_params)
        self.ids = {i: imgid for i, imgid in enumerate(img_ids)}
        self.rev_ids = {v: k for k, v in self.ids.items()}
        self.transforms_are_albumentation = isinstance(self.transforms, BaseCompose)
        self.convert_rgb = convert_rgb
        self.preprocessing = preprocessing

    def apply_filter(self, img_ids, **filter_params):
        # place holder for now
        return img_ids

    @staticmethod
    def read_image(filename):

        image = sitk.GetArrayFromImage(sitk.ReadImage(filename)).squeeze().astype('float32')


        return image

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is a list of captions for the image.
        """
        img_id = self.ids[index]
        image_row = self.data.loc[img_id]
        path = image_row['filepath']

        img = self.read_image(os.path.join(self.root, path))
        target = [(image_row[self.label_map[c]]).astype('float32') for c in self.class_order]

        output = dict(image=img)
        if self.transforms is not None:
            if self.transforms_are_albumentation:
                output = self.transforms(**output)
            else:
                raise NotImplementedError('Not implemented yet, must be albumentation based transform')

        if self.preprocessing:
            target = torch.tensor(target).float()
            output = self.preprocessing(**output)

        output['index'] = index
        output['image_id'] = img_id
        output['target'] = target

        return output

    def __len__(self):
        return len(self.ids)

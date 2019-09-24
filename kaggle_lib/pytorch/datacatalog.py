import os

from .datasets import RSNA2019Dataset

dataset_map = {
    'rsna2019-stage1': {
        'train': 'rsna2019-stage1-train',
        'val': None,
        'test': 'rsna2019-stage1-test',
        'classes': ['any', 'edh', 'iph', 'ivh', 'sah', 'sdh']},

}

datacatalog = {
    'rsna2019-stage1-train': {
        "img_dir": 'rsna-intracranial-hemorrhage-detection/stage_1_train_images/',
        'csv_file': 'rsna-intracranial-hemorrhage-detection/stage_1_train_with_headers.csv'
    },

    'rsna2019-stage1-test': {
        "img_dir": 'rsna-intracranial-hemorrhage-detection/stage_1_test_images/',
        'csv_file': 'rsna-intracranial-hemorrhage-detection/stage1-test-headers.csv'
    },

    None: {}

}


def get_csv_file(dataset_dict, data_root):
    if dataset_dict:
        return os.path.join(data_root, dataset_dict['csv_file'])


def get_dataset(dataset_dict, data_root, **kwargs):
    if dataset_dict:
        root = os.path.join(data_root, dataset_dict['img_dir'])
        csv_file = os.path.join(data_root, dataset_dict['csv_file'])
        return RSNA2019Dataset(root=root, csv_file=csv_file, **kwargs)

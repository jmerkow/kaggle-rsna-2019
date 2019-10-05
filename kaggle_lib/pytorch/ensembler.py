# from .crf import dense_crf
import csv
import hashlib
import json
import os
from collections import defaultdict

import albumentations as A
import numpy as np
import pandas as pd
import six
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from kaggle_lib.pytorch.augmentation import get_preprocessing, make_transforms
from kaggle_lib.pytorch.datacatalog import get_dataset, dataset_map, datacatalog
from kaggle_lib.pytorch.get_model import get_model
from kaggle_lib.pytorch.tta import TTATransform
from kaggle_lib.pytorch.utils import HidePrints, Timer, hms_string


def make_submission_df(raw_csv_file, clip=0):
    raw_df = pd.read_csv(raw_csv_file).set_index('ID')
    submission_df = raw_df.unstack()

    submission_df.index.names = ['label_name', 'ID']
    submission_df.name = 'Label'

    submission_df = submission_df.swaplevel(0, 1).sort_index().to_frame().reset_index()
    submission_df['ID'] = submission_df['ID'] + '_' + submission_df['label_name']

    submission_df = submission_df.drop_duplicates(subset=['ID'])
    submission_df = submission_df.set_index(['ID'])[['Label']]

    if clip:
        submission_df['Label'] = submission_df['Label'].apply(lambda x: np.clip(x, clip, 1 - clip))

    return submission_df

def is_model_dict(d):
    return 'model_fn' in d


def make_model_name(model_name, tta_function=None):
    bad_names = ['mnt', 'nas', 'experiments-kaggle', 'rsna2019', 'checkpoints', 'results']
    model_name = '_'.join([s for s in model_name.strip(os.sep, ).split(os.sep) if s not in bad_names]).replace(
        '.pth.tar', '')
    return '{}-tta{}'.format(model_name, str(tta_function).title())


def is_model_dict(d):
    return 'model_fn' in d


def get_ensemble_name(dirs):
    hasher = hashlib.md5()
    for fn in dirs:
        hasher.update(fn.encode())
    h = hasher.hexdigest()[:6]
    return str(h)


def _make_model_dict(models, **extras):
    if isinstance(models, dict):
        models = [{name: m} for name, m in models.items()]
    names, models = zip(*[_get_name_and_models(model, **extras) for model in models])
    return {get_ensemble_name(sorted(names)): dict(zip(names, models))}


def _get_name_and_models(model, **extras):
    if isinstance(model, six.string_types):
        return make_model_name(model, **extras), dict(model_fn=model, **extras)

    if isinstance(model, dict):
        return list(model.keys())[0], list(model.values())[0]

    model = _make_model_dict(model, **extras)
    return list(model.keys())[0], list(model.values())[0]


def drop_duplicate_index(df):
    # Fastest index drop:
    # https://stackoverflow.com/questions/13035764/remove-rows-with-duplicate-indices-pandas-dataframe-and-timeseries/34297689#34297689
    return df[~df.index.duplicated(keep='first')]


def make_model_dict(models, name=None, **extras):
    if isinstance(models, dict) and len(models) == 1:
        name, models = list(models.items())[0]

    name_hash, models = list(_make_model_dict(models, **extras).items())[0]

    if name is None:
        name = name_hash

    return name, models


class Ensembler(object):
    label_map = {
        'any': 'any',
        'edh': 'epidural',
        'iph': 'intraparenchymal',
        'ivh': 'intraventricular',
        'sah': 'subarachnoid',
        'sdh': 'subdural',
    }

    __repr_indent__ = 3

    def __init__(self, model_root,
                 data_root='/data',
                 output_dir=None,
                 allow_skip=True,
                 mode='test',
                 dataset='rsna2019-stage1',
                 n_jobs=0):

        self.timers = defaultdict(Timer)

        self.model_root = model_root
        if output_dir is None:
            output_dir = model_root
        self.output_dir = output_dir
        self.mode = mode
        self.dataset = dataset
        self.data_root = data_root

        self.kaggle_ids = list(self.get_dataset().ids.values())
        self.allow_skip = allow_skip

        self.n_jobs = n_jobs

        self.name = None
        self.results = []

        self.pred_df = None
        self.submit_df = None
        self.tta_type = None

        self.reset_counts()

    def _repr_preamble_(self):
        head = "Dataset " + self.dataname
        body = ["Number of datapoints: {}".format(len(self.kaggle_ids))]
        if self.model_root is not None:
            body.append("Model Root: {}".format(self.model_root))
        if self.output_dir is not None:
            body.append("Output directory: {}".format(self.output_dir))
        lines = [head] + [" " * self.__repr_indent__ + line for line in body]

        return '\n'.join(lines)

    def _repr_ensemble_(self):
        head = "Ensemble " + self.name
        body = ["TTA: {} ".format(self.tta_type)]

        body.append("Models")
        body.extend([" " * self.__repr_indent__ + l for l in json.dumps(self.models, indent=2).split('\n')])

        body.append("Models Dict")
        body.extend([" " * self.__repr_indent__ + l for l in json.dumps(self.models_dict, indent=2).split('\n')])

        lines = [head] + [" " * self.__repr_indent__ + line for line in body]

        return '\n'.join(lines)

    def __repr__(self):
        parts = [self._repr_preamble_()]
        if self.name:
            parts.append(self._repr_ensemble_())
        return '\n'.join(parts)

    def __str__(self):
        return self.__repr__()

    def reset_counts(self):
        self.n_infers = 0
        self.n_ensembles = 0
        self.iskip = 0
        self.icompl = 0
        self.eskip = 0
        self.ecompl = 0

    def get_dataset(self, transforms=None, preprocessing=None, **kwargs):
        return get_dataset(self.datacatalog, self.data_root, transforms=transforms,
                           preprocessing=preprocessing, reader='dcm', **kwargs)

    def get_dataloader(self, batch_size=1, hide_print=True, **kwargs):
        if hide_print:
            with HidePrints():
                ds = self.get_dataset(**kwargs)
        else:
            ds = self.get_dataset(**kwargs)
        return DataLoader(ds, shuffle=False, batch_size=batch_size, drop_last=False)

    @property
    def datacatalog(self):
        return datacatalog[dataset_map[self.dataset][self.mode]]

    @staticmethod
    def load_model(checkpoint):
        model_params = checkpoint['model_params']
        model_params.pop('weights', None)
        model, preprocessing = get_model(**model_params)
        model.load_state_dict(checkpoint['state_dict'])
        transforms = A.from_dict(checkpoint['test_transforms'])
        return model, preprocessing, transforms

    @staticmethod
    def get_written_ids(save_full_filename):
        df = pd.read_csv(save_full_filename)
        return df['ID'].unique().tolist()

    @property
    def dataname(self):
        return '_'.join([self.dataset, self.mode])

    def write_model_results(self, model_fn, batch_size=8, desc='', force_overwrite=False):
        checkpoint = torch.load(os.path.join(self.model_root, model_fn),
                                map_location='cuda:{}'.format(torch.cuda.current_device()))
        model, preprocessing, transforms = self.load_model(checkpoint)

        epoch = checkpoint['epoch']
        class_order = checkpoint['class_order']

        job_dir = os.path.dirname(os.path.dirname(model_fn))

        transform_params = checkpoint.get('transform', None)
        if transform_params is None:
            # get them from the yaml...
            with open(os.path.join(self.model_root, job_dir, 'train.yml')) as f:
                config = yaml.safe_load(f)
            transform_params = config['cfg']['data']

        augmentation = transform_params['augmentation']
        data_shape = transform_params['data_shape']

        transforms = make_transforms(data_shape, **augmentation, apply_crop=True)
        tta_transform = TTATransform(data_shape=data_shape, **self.tta_params)
        assert tta_transform.name == self.tta_type, 'ttas do not match'
        save_filename = "results/{dataname}/model_step{epoch:0>3}_TTA-{tta_type}.csv".format(dataname=self.dataname,
                                                                                             epoch=epoch,
                                                                                             tta_type=self.tta_type)
        save_full_filename = os.path.join(self.output_dir, job_dir, save_filename)
        write_dir = os.path.dirname(save_full_filename)

        print("[Infer {}]".format(desc), "save_full_filename:", save_full_filename, "force_overwrite:", force_overwrite)

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        if not os.path.exists(save_full_filename) or force_overwrite:
            new_file = True
            need_write = self.kaggle_ids[:]
        else:
            new_file = False
            already_written_ids = self.get_written_ids(save_full_filename)
            need_write = [kid for kid in self.kaggle_ids if kid not in already_written_ids]

        if not len(need_write):
            print("[Infer {}]".format(desc), "Already Completed! count:", len(already_written_ids))
            self.iskip += 1
            return save_full_filename

        print("[Infer {}]".format(desc), "num kaggle ids:", len(self.kaggle_ids), "Need Write:", len(need_write))
        print("[Infer {}]".format(desc), 'Model:', model_fn)

        model = model.cuda()
        model.eval()
        loader = self.get_dataloader(transforms=transforms,
                                     preprocessing=get_preprocessing(preprocessing, data_shape=data_shape),
                                     batch_size=batch_size, image_ids=need_write, tta_transform=tta_transform)

        pbar = tqdm.tqdm(enumerate(loader), total=len(loader), desc="Infer [{}]".format(desc))
        skipped = 0

        file_mode = 'a' if not new_file else 'w'
        with open(save_full_filename, mode=file_mode) as f:
            writer = csv.DictWriter(f, fieldnames=['ID'] + list(self.label_map.values()))
            if new_file:
                writer.writeheader()
            for i, sample in pbar:
                pbar.set_postfix_str("Skipped: {}".format(skipped))

                kaggle_ids = sample['image_id']
                image = sample['image'].cuda()
                if image.ndim > 4:
                    bs, ncrops, c, h, w = image.size()
                    scores = model.predict(image.view(-1, c, h, w))
                    scores = scores.view(bs, ncrops, -1).mean(1)
                else:
                    scores = model.predict(image)
                scores = scores.cpu().detach().numpy()
                for n, ImageId in enumerate(kaggle_ids):
                    row = {
                        'ID': ImageId,
                    }

                    row.update({self.label_map[c]: r for c, r in zip(class_order, scores[n])})
                    writer.writerow(row)
        self.icompl += 1
        return save_full_filename

    def run_ensemble(self, models, name, batch_size=1, desc=''):

        self.timers[name].tic()
        if not desc:
            desc = name
        scorecards = []
        count = len(models)
        for i, (name_, mdl) in enumerate(models.items(), 1):
            if is_model_dict(mdl):
                self.timers['inference'].tic()
                scorecards.append(
                    self.write_model_results(batch_size=batch_size, desc="{}:({}/{}) i".format(desc, i, count), **mdl))
                self.n_infers += 1
                self.timers['inference'].toc()
            else:

                scorecards.append(self.run_ensemble(mdl, batch_size=batch_size, name=name_,
                                                    desc="{}:({}/{}) {}".format(desc, i, count, name_, )))
                self.n_ensembles += 1

        self.timers['ensemble'].tic()
        scorecard = self.ensemble_dirs(scorecards, name=name, desc=desc, models=models)
        self.timers['ensemble'].toc()
        self.timers[name].toc()
        print("[Ensemble {}]".format(desc),
              'Completed ensemble {} took {}. Elapsed since start {}.'.format(name,
                                                                              hms_string(self.timers[name].total_time),
                                                                              hms_string(
                                                                                  self.timers['start'].elapsed_time)))
        print()
        return scorecard

    def ensemble_dirs(self, scorecards, name, models, desc=''):

        job_dir = "ensembles/ensemble-{name}".format(name=name)

        save_filename = "results/{dataname}/model_step{epoch}_TTA-{tta_type}.csv".format(dataname=self.dataname,
                                                                                         epoch="None",
                                                                                         tta_type=self.tta_type)
        save_full_filename = os.path.join(self.output_dir, job_dir, save_filename)
        write_dir = os.path.dirname(save_full_filename)
        models_file = os.path.join(self.output_dir, job_dir, 'models.json')

        if not os.path.exists(write_dir):
            os.makedirs(write_dir)

        if os.path.exists(models_file):
            with open(models_file, 'r') as f:
                models_ = json.load(f)
            assert models_ == models
        else:
            with open(models_file, 'w') as f:
                json.dump(models, f, indent=2)

        dfj = []
        pbar = tqdm.tqdm(scorecards, desc="Ensemble [{}]".format(desc))
        for card in pbar:
            df = pd.read_csv(card).set_index('ID')
            df = drop_duplicate_index(df)
            assert len(df) == len(self.kaggle_ids)
            dfj.append(df)
        dfj = pd.concat(dfj, keys=range(len(dfj)), axis=0)

        print(len(dfj), len(dfj) / len(models))

        scorecard = dfj.groupby('ID').mean()
        if not os.path.exists(write_dir):
            os.makedirs(write_dir)
        scorecard.to_csv(save_full_filename)
        return save_full_filename

    def ensemble(self, models, name=None, batch_size=8, **tta_params):

        name_, models_dict = make_model_dict(models)

        self.models = models
        self.models_dict = models_dict

        if name is None:
            name = name_
        self.name = name
        self.tta_params = tta_params
        self.tta_type = TTATransform(data_shape=[10, 10], **self.tta_params).name

        print(self)

        print()
        print('Starting!')
        self.reset_counts()
        self.timers['start'].tic()
        self.results_file = self.run_ensemble(models_dict, name=self.name, batch_size=batch_size, desc=self.name)
        print('Ensemble Completed!')
        self.timers['start'].toc()
        ensemble_total_time = self.timers['ensemble'].total_time
        ensemble_average_time = self.timers['ensemble'].average_time

        infer_total_time = self.timers['inference'].total_time
        infer_average_time = self.timers['inference'].average_time

        print('Ensemble count: {} (skipped: {}, completed: {}).  Total time: {}, average time: {}'.format(
            self.n_ensembles,
            self.eskip,
            self.ecompl,
            hms_string(ensemble_total_time),
            hms_string(ensemble_average_time)))
        print('Inference count: {} (skipped: {}, completed: {}). Total time: {}, average time: {}'.format(
            self.n_infers,
            self.iskip, self.icompl,
            hms_string(infer_total_time),
            hms_string(infer_average_time)))
        print('Total time: {}'.format(hms_string(self.timers['start'].total_time)))

        #         self.load_results()

        return self.results_file






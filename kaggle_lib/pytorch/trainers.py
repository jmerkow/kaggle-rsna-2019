import logging
import os
import random
import re
import sys
from collections import defaultdict
from copy import deepcopy

import albumentations as A
import numpy as np
import torch
import torch.optim
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchnet.meter import AverageValueMeter
from tqdm import tqdm

# from .datacatalog import dataset_map, datacatalog
from kaggle_lib.pytorch.datasets import get_csv_file, get_dataset, get_data_constants, SequenceBatchSampler
from .augmentation import make_augmentation, make_transforms, get_preprocessing
from .get_model import get_model
from .loss import Criterion
from .lr_scheduler import get_scheduler
from .metrics import RSNA2019Metric
from .optimizers import RAdam, Ralamb, Lookahead
from .saver import Saver
from .summary import TensorboardSummary
from .sync_batchnorm import convert_model, patch_replication_callback
from .utils import get_gpu_ids, StopOnPlateau, rsna2019_split, Timer

logger = logging.getLogger(__name__)


class ClassifierTrainer(object):
    optimizer_types = {'asgd': torch.optim.ASGD,
                       'adadelta': torch.optim.Adadelta,
                       'adagrad': torch.optim.Adagrad,
                       'adam': torch.optim.Adam,
                       'adamx': torch.optim.Adamax,
                       'adamw': torch.optim.AdamW,
                       'lbfgs': torch.optim.LBFGS,
                       'rmsprop': torch.optim.RMSprop,
                       'rprop': torch.optim.Rprop,
                       'sgd': torch.optim.SGD,
                       'sparceadam': torch.optim.SparseAdam,
                       'radam': RAdam,
                       'ralamb': Ralamb,
                       }

    section_defaults = {
        'random': {'seed': None, 'backend_deterministic': False, 'backend_benchmark': False},
        'model': {
            'encoder': 'se_resnext101_32x4d',
            'nclasses': 6,
            'encoder_weights': 'imagenet',
            'activation': 'sigmoid',
            'model_dir': '/data/pretrained_weights/',
            'weights': None,
            'classifier': 'basic',
            'final_output': 'final',
        },
        'data': {
            'dataset': 'rsna2019-stage1',
            'load_options': {},  ## will use this for doing cool stuff with channels etc
            'sampler': None,
            'random_split': True,
            'n_splits': 10,
            'fold': 0,
            'random_split_state': None,
            'random_split_stratified': None,
            'random_split_group': 'sop_instance_uid',
            'batch_size': 32,
            'max_images_per_card': None,
            'num_workers': 2,
            'data_shape': (224, 224),
            'epochs': 15,
            'data_root': '/data/',
            'augmentation': {'resize': 'auto'},
            'classes': ('sdh', 'sah', 'ivh', 'iph', 'edh', 'any'),
            'filter': {},
            'reader': 'h5',
            'pin_memory': True,
            'extra_datasets': None,
            'sequence_mode': False,

        },
        'optimization': {
            'optimizer': {'type': 'adam', },
            'base_lr': 1e-5,
            'classifier_lr_mult': 10,
            'lr_scheduler': {'step_on_iter': False, 'scale_params': False, 'pass_epoch': False},
            'norm_loss_for_step': True,
            'plateau_scheduler': None,
            'loss': {'type': 'bce', 'loss_weights': (.142857, .142857, .142857, .142857, .142857, 0.285715)},
            'use_lookahead': False,
            'lookahead_params': None,
            'stopper': None,
        },

        'metric': {
            'best_metric': 'loss',
            'score_bigger_is_better': False,
            'loss_weights': (.142857, .142857, .142857, .142857, .142857, 0.285715)

        },
        'other': {
            'train_log_every': 10,
            'sync_bn': False,
            'visual_count': 8,
            'visualize_every': 50,
        }
    }

    model_filename_format = "model_step{:03d}.pth.tar"
    model_glob_str = 'model_step*.pth.tar'
    model_epoch_regex = 'model_step(?P<step>\d*).pth.tar'
    model_subdir = 'checkpoints'

    def _is_metric_better(self, curr_score):
        if self.prev_score is None:
            return True
        if self.score_bigger_is_better:
            return curr_score > self.prev_score
        else:
            return curr_score < self.prev_score

    def __init__(self, workdir='.', **kwargs):
        self.prev_score = None
        self.timers = defaultdict(Timer)
        self.train_timers = defaultdict(Timer)

        self.workdir = os.path.abspath(workdir)

        if not os.path.exists(self.workdir):
            os.makedirs(self.workdir)

            logger.info("Workdir: %s", self.workdir)
        self.summary = TensorboardSummary(self.workdir)
        self.saver = Saver(self.workdir, subdir=self.model_subdir)
        self.config = {}
        for k, section in self.section_defaults.items():
            self.config[k] = {}
            self.config[k].update(section.copy())
            self.config[k].update(kwargs.get(k, {}))
            setattr(self, k + '_params', self.config[k])

        logger.info(yaml.safe_dump(self.config))
        self.setup()

        self.start_epoch = 0
        self.try_resume()

    def try_resume(self):
        fs = self.saver.get_checkpoints(self.model_glob_str)
        if len(fs):
            files_dict = {}
            for f in fs:
                match = re.search(self.model_epoch_regex, f)
                if match:
                    files_dict[int(match.groupdict()['step'])] = f

            max_step = max(files_dict.keys())
            latest_checkpoint_path = files_dict[max_step]

            logger.info('resuming at step: %d, path: %s', max_step, latest_checkpoint_path)
            self.load_checkpoint(latest_checkpoint_path)

    def setup(self):
        self.timers['setup'].tic()
        self._setup_random(**deepcopy(self.random_params))
        self._setup_model(**deepcopy(self.model_params))
        self._setup_data(**deepcopy(self.data_params))
        self._setup_optimization(**deepcopy(self.optimization_params))
        self._setup_metrics(**deepcopy(self.metric_params))
        self._setup_other(**deepcopy(self.other_params))
        self.timers['setup'].toc()
        logger.info('Setup took %s', self.timers['setup'].total_time_str)

    def _setup_random(self, seed=None, backend_deterministic=False, backend_benchmark=False):
        if seed is not None:
            random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = backend_deterministic
        torch.backends.cudnn.benchmark = backend_benchmark

    def _setup_model(self, weights=None, **model_params):
        logger.info('====MODEL====')

        with open(os.path.join(self.workdir, 'model_params.yml'), 'w') as f:
            print(yaml.safe_dump(model_params), file=f)

        self.final_output = model_params.pop('final_output')
        self.model, self.model_preprocessing = get_model(**model_params)

        self.required_inputs = self.model.required_inputs


        if weights:
            model_dir = model_params['model_dir']
            weights_file = os.path.join(model_dir, weights)
            if not os.path.isfile(weights_file):
                raise RuntimeError("=> no checkpoint found at '{}'".format(weights_file))
            logger.info("loading weigths from %s", weights_file)
            checkpoint = torch.load(weights_file)
            self.model.load_state_dict(checkpoint['state_dict'])

        self.is_multi_task = self.model.is_multi_task

        logger.info("model: %s", self.model.name)
        logger.info("model output info: %s", self.model.output_info())

    def _setup_data(self, **data_params):

        self.data_shape = data_params['data_shape']
        augmentation = data_params['augmentation']
        self.dataset = dataset = data_params['dataset']
        self.epochs = data_params['epochs']
        self.data_root = data_params['data_root']
        self.classes = data_params['classes']
        filter_params = data_params['filter']
        num_workers = data_params['num_workers']

        self.sequence_mode = data_params['sequence_mode']

        random_split = data_params['random_split']
        n_splits = data_params['n_splits']
        fold = data_params['fold']
        random_split_group = data_params['random_split_group']
        random_split_state = data_params['random_split_state']
        random_split_stratified = data_params['random_split_stratified']

        extra_datasets = data_params.pop('extra_datasets', None)

        reader = data_params['reader']
        val_reader = data_params.get('reader', reader)

        pin_memory = data_params['pin_memory']

        self.batch_size = data_params['batch_size']
        self.max_images_per_card = data_params.get('max_images_per_card', None)
        self.device_ids = get_gpu_ids()

        self.step_size = 1
        if self.max_images_per_card is not None:
            num_cards = len(self.device_ids)
            images_per_iteration = self.max_images_per_card * num_cards
            self.step_size, rem = divmod(self.batch_size, images_per_iteration)
            assert rem == 0, "batch size={} does not evenly fit on {} cards with max_images_per_card: {}".format(
                self.batch_size, num_cards, self.max_images_per_card
            )

        logger.info('====BATCH_INFO====')
        batch_size = int(self.batch_size / self.step_size)
        logger.info('Batch Size %d, step_size: %d, Data Batch Size: %d', self.batch_size, self.step_size, batch_size)

        self.train_transforms = make_augmentation(self.data_shape, **augmentation)
        self.val_transforms = make_transforms(self.data_shape, **augmentation)
        self.test_transforms = make_transforms(self.data_shape, **augmentation)
        self.transform_args = dict(data_shape=self.data_shape, augmentation=augmentation)

        datacatalog, dataset_map = get_data_constants(self.data_root)
        train_dataset_name, val_dataset_name = dataset_map[self.dataset]['train'], dataset_map[dataset]['val']

        train_catalog = datacatalog[train_dataset_name]
        val_catalog = datacatalog[val_dataset_name]

        train_img_ids = None
        val_img_ids = None

        if random_split:
            if val_catalog is not None:
                ValueError('You are using a dataset which has already been split!')
            val_catalog = train_catalog
            csv_file = get_csv_file(train_catalog, self.data_root)

            train_img_ids, val_img_ids = rsna2019_split(csv_file=csv_file, group_on=random_split_group,
                                                        n_splits=n_splits, fold=fold,
                                                        random_state=random_split_state,
                                                        stratified=random_split_stratified)
            logger.info(
                "Using Random Split! n_splits: %d, fold: %d, random_state: %s", n_splits, fold, str(random_split_state))

        apply_filter_to_val = filter_params.pop('apply_to_val', True)

        if apply_filter_to_val:
            val_filter_params = deepcopy(filter_params)
        else:
            val_filter_params = {}

        train_dataset = get_dataset(train_catalog, self.data_root, transforms=self.train_transforms,
                                    preprocessing=get_preprocessing(self.model_preprocessing),
                                    img_ids=train_img_ids, class_order=self.classes, reader=reader,
                                    extra_fields=self.required_inputs,
                                    extra_datasets=extra_datasets,
                                    **filter_params)
        val_dataset = get_dataset(val_catalog, self.data_root, transforms=self.val_transforms,
                                  preprocessing=get_preprocessing(self.model_preprocessing),
                                  img_ids=val_img_ids, class_order=self.classes, reader=val_reader,
                                  extra_fields=self.required_inputs,
                                  **val_filter_params)

        logger.info("Num Images, train: %d, val: %d", len(train_dataset), len(val_dataset))

        with open(os.path.join(self.workdir, 'training_images.yml'), 'w') as f:
            print(yaml.safe_dump({'train': list(train_dataset.ids.values()),
                                  'val': list(val_dataset.ids.values())}), file=f)

        logger.info('====DATA====')
        logger.info('TRAINING')
        logger.info(str(train_dataset))

        logger.info('VALIDATION')
        logger.info(str(val_dataset))

        self.val_loader = None
        if not self.sequence_mode:
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                           num_workers=num_workers, pin_memory=pin_memory)
            if val_dataset is not None:
                self.val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, drop_last=False,
                                             pin_memory=pin_memory)
        else:
            self.train_loader = DataLoader(train_dataset,
                                           batch_sampler=SequenceBatchSampler(train_dataset, shuffle=True),
                                           num_workers=num_workers, pin_memory=pin_memory)
            if val_dataset is not None:
                self.val_loader = DataLoader(val_dataset,
                                             batch_sampler=SequenceBatchSampler(val_dataset, shuffle=False),
                                             num_workers=4, pin_memory=pin_memory)

        self.iters_in_epoch = len(self.train_loader)

    def _setup_optimization(self, **optimization_params):
        logger.info('====OPTIMIZATION====')
        base_lr = optimization_params.pop('base_lr')
        classifier_lr_mult = optimization_params.pop('classifier_lr_mult')

        optimizer = optimization_params.pop('optimizer')
        optimizer_type = optimizer.pop('type')
        optimizer_params = optimizer.pop('optimizer_params', {})

        lr_scheduler_params_ = optimization_params.pop('lr_scheduler', {}) or {}
        plateau_scheduler_params = optimization_params.pop('plateau_scheduler', None)

        stop_on_plateau_params = optimization_params.pop('stopper', None)
        loss_params = optimization_params.pop('loss')

        self.criterion = Criterion(classes=self.classes, final_output=self.final_output, **loss_params)

        self.norm_loss_for_step = optimization_params.pop('norm_loss_for_step', True)

        assert optimizer_type in self.optimizer_types, 'un supported optimizer type'
        params = [
            {'params': self.model.get_encoder_params(), 'lr': base_lr},
            {'params': self.model.get_classifier_params(), 'lr': base_lr * classifier_lr_mult}

        ]

        optimizer = self.optimizer_types[optimizer_type](params, **optimizer_params)
        use_lookahead = optimization_params.pop('use_lookahead')

        if use_lookahead:
            logger.info("Using lookahead optimization!")
            lookahead_params = optimization_params.pop('lookahead_params') or {}
            self.optimizer = Lookahead(optimizer, **lookahead_params)
        else:
            self.optimizer = optimizer

        self.plateau_scheduler = None
        plateau_scheduler_str = "None"
        if plateau_scheduler_params:
            self.plateau_scheduler = ReduceLROnPlateau(self.optimizer, **plateau_scheduler_params)
            plateau_scheduler_str = ', '.join("{}={}".format(k, v)
                                              for k, v in self.plateau_scheduler.state_dict().items())

        self.lr_scheduler = None
        self.scheduler_pass_epoch = None
        if lr_scheduler_params_ is not None:
            self.scheduler_pass_epoch = lr_scheduler_params_.pop('pass_epoch', False)
            lr_scheduler_params = {}
            if 'schedulers' not in lr_scheduler_params_:
                lr_scheduler_params['schedulers'] = lr_scheduler_params_
            else:
                lr_scheduler_params = lr_scheduler_params_

            self.lr_step_on_iter = lr_scheduler_params.pop('step_on_iter', False)
            if self.lr_step_on_iter:
                self.steps_per_epoch = self.iters_in_epoch
            else:
                self.steps_per_epoch = 1
            self.lr_scheduler = get_scheduler(self.optimizer,
                                              steps_per_epoch=self.steps_per_epoch,
                                              **lr_scheduler_params
                                              )

        self.stopper = None
        if stop_on_plateau_params is not None:
            self.stopper = StopOnPlateau(**stop_on_plateau_params)
        stopper_str = ', '.join(
            "{}={}".format(k, v) for k, v in self.stopper.state_dict().items()) if self.stopper else "None"

        logger.info('OPTIMIZER')
        logger.info(str(self.optimizer))

        logger.info('SCHEDULERS')
        logger.info(str(self.lr_scheduler))
        logger.info("Plateau LR Reduce: %s", str(plateau_scheduler_str))

        logger.info("LOSS")
        logger.info(str(self.criterion))
        logger.info("norm_loss_for_step: %s", str(self.norm_loss_for_step))

        logger.info("OTHER")
        logger.info("Stopper: %s", str(stopper_str))

    def _setup_metrics(self, **metric_params):
        logger.info('====METRICS====')
        self.score_bigger_is_better = metric_params.pop('score_bigger_is_better', False)
        self.best_metric = metric_params.pop('best_metric')
        self.val_metrics = RSNA2019Metric(classes=self.classes, final_output=self.final_output, **metric_params)

    def _setup_other(self, **other_params):
        logger.info('====OTHER====')
        self.train_log_every = min(other_params['train_log_every'], len(self.train_loader))
        self.visualize_every = min(other_params['visualize_every'], len(self.train_loader))
        self.visual_count = other_params['visual_count']
        self.sync_bn = other_params['sync_bn']

        logger.info("train_log_every: %s", str(self.train_log_every))
        logger.info("visualize_every: %s", str(self.visualize_every))
        logger.info("sync_bn: %s", str(self.sync_bn))

    def _format_logs(self, logs):
        str_logs = ['{}={:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def step_epoch(self, epoch):

        logs = {}
        epoch_meters = defaultdict(lambda: AverageValueMeter())
        intra_epoch_meters = defaultdict(lambda: AverageValueMeter())
        self.model.train()

        self.optimizer.zero_grad()
        last_step = 0
        num_img_tr = len(self.train_loader)
        tdl = iter(self.train_loader)
        tbar = tqdm(list(range(num_img_tr)), desc='Epoch {} Train'.format(epoch), file=sys.stdout)
        local_timers = self.train_timers

        for i in tbar:
            local_timers['iter'].tic()
            local_timers['data'].tic()
            sample = next(tdl)
            local_timers['data'].toc()

            step = i + num_img_tr * epoch
            metrics = {}
            for lri, lr in enumerate([p['lr'] for p in self.optimizer.param_groups]):
                metrics['lr-{}'.format(lri)] = lr

            local_timers['to_cuda'].tic()
            image = sample.pop('image').cuda()
            target = sample.pop('target').cuda()
            extra_inputs = {k: sample[k].cuda() for k in self.required_inputs}
            local_timers['to_cuda'].toc()

            local_timers['score'].tic()
            scores = self.model(image, **extra_inputs)
            local_timers['score'].toc()

            local_timers['loss'].tic()
            metrics.update(self.criterion(scores, target))
            local_timers['loss'].toc()

            local_timers['backward'].tic()
            self.criterion.backward()
            local_timers['backward'].toc()

            if (i + 1) % self.step_size == 0:
                local_timers['step'].tic()
                last_step = i
                epoch_dec = epoch + i / self.steps_per_epoch
                self.optimizer.step()
                self.optimizer.zero_grad()
                if self.lr_scheduler is not None and self.lr_step_on_iter:
                    step_epoch = epoch_dec if self.scheduler_pass_epoch else None
                    self.lr_scheduler.step(step_epoch)
                local_timers['step'].toc()

            local_timers['iter'].toc()
            metrics.update({'time-' + name: t.diff for name, t in local_timers.items()})

            for name, value in metrics.items():
                epoch_meters[name].add(value)
                intra_epoch_meters[name].add(value)
                logs[name] = epoch_meters[name].mean
                if (i + 1) % self.train_log_every == 0:
                    write_name = 'train/{}'.format(name) \
                        if not name.startswith('time-') else 'timer/train/{}'.format(name.replace('time-', ''))

                    self.summary.add_scalar(write_name, intra_epoch_meters[name].mean, step)
                    intra_epoch_meters[name].reset()

            s = self._format_logs(logs)
            tbar.set_postfix_str(s)

        if last_step != i:
            self.optimizer.step()
            self.optimizer.zero_grad()
            if self.lr_scheduler is not None and self.lr_step_on_iter:
                step_epoch = epoch if self.scheduler_pass_epoch else None
                self.lr_scheduler.step(step_epoch)

        for name, value in logs.items():
            write_name = 'train-epoch/mean-{}'.format(name) \
                if not name.startswith('time-') else 'timer-epoch/train/mean-{}'.format(name.replace('time-', ''))
            self.summary.add_scalar(write_name, value, epoch + 1)
        t_logs = {}
        for name, t in local_timers.items():
            t_logs['timer-total-{}'.format(name)] = t.total_time_str
            self.summary.add_scalar('timer-epoch/total-{}'.format(name), t.total_time, epoch + 1)
            t.reset()
            # self.summary.add_scalar('train-epoch/timer-average-{}'.format(name), t.average_time, epoch + 1)

        t_s = ', '.join('{}={}'.format(k, v) for k, v in t_logs.items())
        s = self._format_logs(logs)
        logger.info('[Epoch: %d, numImages: %5d] Train scores: %s, times: %s', epoch + 1,
                    i * self.step_size + image.data.shape[0],
                    s, t_s)
        if self.lr_scheduler is not None and not self.lr_step_on_iter:
            step_epoch = epoch if self.scheduler_pass_epoch else None
            self.lr_scheduler.step(step_epoch)
        return logs

    def validation(self, epoch):

        tbar = tqdm(self.val_loader, desc='Epoch {} Validate'.format(epoch), file=sys.stdout)
        self.val_metrics.reset()
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(tbar):
                image = sample.pop('image').cuda()
                target = sample.pop('target').cuda()
                extra_inputs = {k: sample[k].cuda() for k in self.required_inputs}
                scores = self.model(image, **extra_inputs)
                # loss = self.criterion(scores, target)
                # raw_loss = self.criterion.raw_loss
                # metrics['loss'] = loss.cpu().detach().numpy()
                # metrics.update({'loss-' + name: l.cpu().numpy() for name, l in zip(self.classes, raw_loss)})
                self.val_metrics.add_batch(scores, target, sample=sample)

                # for name, value in metrics.items():
                #     meters[name].add(value)
                #     logs[name] = meters[name].mean
                # s = self._format_logs(logs)
                # tbar.set_postfix_str(s)
        logs = self.val_metrics.mean
        for name, value in logs.items():
            self.summary.add_scalar('val-epoch/{}'.format(name), value, epoch + 1)

        s = self._format_logs(logs)
        logger.info('[Epoch: %d, numImages: %5d, Thresholds: %d] Validation scores: %s', epoch + 1,
                    i + 1, len(self.val_metrics.rows), s)
        logs['scorecard'] = self.val_metrics.get_rows()
        # logs['best'] = best
        self.val_metrics.reset()
        return logs

    def run(self, epochs=None):

        epochs = self.epochs or epochs
        self.epochs = epochs

        if self.sync_bn:
            logger.info('syncing batch norm!')
            self.model = torch.nn.DataParallel(convert_model(self.model), device_ids=self.device_ids)
            patch_replication_callback(self.model)
        else:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.device_ids)
        self.model = self.model.cuda()

        for epoch in range(self.start_epoch, self.epochs):
            train_logs = self.step_epoch(epoch)
            val_logs = self.validation(epoch)
            curr_score = val_logs[self.best_metric]

            if self.prev_score is None:
                logger.info("Prev Best Score: None, Current Score: %.5f", curr_score)
                is_best = True
            else:
                logger.info("Prev Best Score: %.5f, Current Score: %.5f", self.prev_score, curr_score)
                is_best = self._is_metric_better(curr_score)
            self.prev_score = curr_score
            self.save_checkpoint(epoch + 1, curr_score, is_best, val_logs, train_logs)

            val_loss = val_logs['loss']
            if self.plateau_scheduler is not None:
                self.plateau_scheduler.step(val_loss)
            if self.stopper is not None:
                self.stopper.step(val_loss)
                if self.stopper.stop:
                    break

        return train_logs, val_logs

    # """
    # class Saver(object):
    #
    #     def __init__(self, workdir='.', subdir='checkpoints'):
    #         self.workdir = os.path.abspath(workdir)
    #         self.directory = os.path.join(self.workdir, subdir)
    #         logger.info("saver at '%s'", self.directory)
    #         if not os.path.exists(self.directory):
    #             os.makedirs(self.directory)
    #
    #     def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=False, **state):
    #         """Saves checkpoint to disk"""
    #         filename = os.path.join(self.directory, filename)
    #         torch.save(state, filename)
    #         logger.info("saving to %s", filename.replace(self.workdir, '').strip('/'))
    #         if is_best:
    #             best_filename = os.path.join(self.directory, 'best_model.pth.tar')
    #             shutil.copy(filename, best_filename, follow_symlinks=False)
    #             logger.info("Best Model! Copying to '%s'", best_filename.replace(self.workdir, '').strip('/'))
    #             """

    def state_dict(self, epoch):
        state = {'epoch': epoch,
                 'state_dict': self.model.module.state_dict(),
                 'optimizer': self.optimizer.state_dict(),
                 'score_metric': self.best_metric,
                 'test_transforms': A.to_dict(self.test_transforms),
                 'transform': self.transform_args,
                 'model_params': self.model_params,
                 'class_order': self.classes,
                 'final_output': self.final_output,
                 }

        if self.lr_scheduler is not None:
            state['lr_scheduler'] = self.lr_scheduler.state_dict()

        if self.plateau_scheduler is not None:
            state['plateau_scheduler'] = self.plateau_scheduler.state_dict()

        if self.stopper is not None:
            state['stopper'] = self.stopper.state_dict()

        return state


    def save_checkpoint(self, epoch, curr_score, is_best, val_logs, train_logs):
        if is_best:
            logger.info('Best model!')
        scorecard = val_logs.pop('scorecard', [])
        state = self.state_dict(epoch=epoch)
        self.saver.save_checkpoint(filename=self.model_filename_format.format(epoch),
                                   is_best=is_best,
                                   score=curr_score,
                                   val_metrics=val_logs,
                                   scorecard=scorecard,
                                   train_metrics=train_logs,
                                   **state)

    def load_checkpoint(self, checkpoint_path):

        state = torch.load(checkpoint_path)

        self.start_epoch = state['epoch']
        self.prev_score = state['score']
        self.model.load_state_dict(state['state_dict'])
        self.optimizer.load_state_dict(state['optimizer'])

        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(state['lr_scheduler'])

        if self.plateau_scheduler is not None:
            self.plateau_scheduler.load_state_dict(state['plateau_scheduler'])

        if self.stopper is not None:
            self.stopper.load_state_dict(state['stopper'])

    def lr_sim(self):
        print('WARNING!! you need to re-create this trainer, running this messed with your LR rates')
        self.model = None
        o = [{k: pa[k] for k in pa if k != 'params'} for pa in self.optimizer.param_groups]

        iters = []
        lrs = []
        epoch_decs = []
        iter_ = 0
        for epoch in range(self.start_epoch, self.epochs):
            for i in range(self.steps_per_epoch):

                epoch_dec = epoch + i / self.steps_per_epoch
                iters.append(iter_)
                epoch_decs.append(epoch_dec)
                lrs.append([p['lr'] for p in self.optimizer.param_groups])

                if (i + 1) % self.step_size == 0:
                    last_step = i
                    step_epoch = epoch_dec if self.scheduler_pass_epoch else None
                    if self.lr_scheduler is not None and self.lr_step_on_iter:
                        self.lr_scheduler.step(step_epoch)
                        
                iter_ += 1

            if not self.lr_step_on_iter:
                step_epoch = step_epoch if self.scheduler_pass_epoch else None
                self.lr_scheduler.step(step_epoch)

        return np.array(iters), np.array(lrs), np.array(epoch_decs)

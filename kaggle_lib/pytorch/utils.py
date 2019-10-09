import os
import sys
import time
from functools import partial

import numpy as np
import pandas as pd
import six
from sklearn.model_selection import StratifiedKFold, KFold
from torch._six import inf
from tqdm import tqdm


def hms_string(total_seconds):
    h = int(total_seconds / (60 * 60))
    m = int((total_seconds % (60 * 60)) / 60)
    s = total_seconds % 60.
    output = []
    if h:
        output.append("{:d}h".format(h))
    if m:
        output.append("{:d}m".format(m))
    output.append("{:.2f}s".format(s))
    return ' '.join(output)


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff

    def reset(self):
        self.start_time = None
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    @staticmethod
    def _to_string(seconds):
        return hms_string(seconds)

    @property
    def total_time_str(self):
        return self._to_string(self.total_time)

    @property
    def average_time_str(self):
        return self._to_string(self.average_time)

    @property
    def diff_str(self):
        return self._to_string(self.diff)

    @property
    def elapsed_time(self):
        if self.start_time is None:
            return -1
        else:
            return time.time() - self.start_time


class HidePrints(object):
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def get_gpu_ids():
    gpu_inds = None
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices:
        gpu_inds = list(map(int, cuda_visible_devices.split(',')))
        assert -1 not in gpu_inds, \
            'Hiding GPU indices using the \'-1\' index is not supported'
    return gpu_inds


def calculate_weigths_labels(dataloader, num_classes):
    # Create an instance from the data loader
    z = np.zeros((num_classes,))
    # Initialize tqdm
    tqdm_batch = tqdm(dataloader)
    print('Calculating classes weights')
    for _, y in tqdm_batch:
        y = y.detach().cpu().numpy()
        mask = (y >= 0) & (y < num_classes)
        labels = y[mask].astype(np.uint8)
        count_l = np.bincount(labels, minlength=num_classes)
        z += count_l
    tqdm_batch.close()
    total_frequency = np.sum(z)
    class_weights = []
    for frequency in z:
        class_weight = 1 / (np.log(1.02 + (frequency / total_frequency)))
        class_weights.append(class_weight)
    ret = np.array(class_weights)

    return ret


class StopOnPlateau(object):

    def __init__(self, mode='min', patience=10,
                 verbose=False, threshold=1e-4, threshold_mode='rel', eps=1e-8):

        self.stop = False
        self.patience = patience
        self.verbose = verbose
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.stop = True

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key not in {'optimizer', 'is_better'}}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)
        self._init_is_better(mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode)


def kfold_split(items, y=None, **options):
    options.setdefault('n_splits', 10)
    if y is None:
        return list(KFold(**options).split(items))
    else:
        return list(StratifiedKFold(**options).split(items, y))


def get_kth_split(items, y=None, fold=0, **options):
    return kfold_split(items, y=y, **options)[fold]


def rsna2019_split(csv_file, group_on='ImageID', stratifier='label__any',
                   stratified=True, **options):
    data = pd.read_csv(csv_file)

    grps = data.groupby(group_on)[stratifier].max()
    items = np.array(grps.index.tolist())
    y = grps.values.tolist() if stratified else None
    train_ix, val_ix = get_kth_split(items, y=y, **options)

    train_ids = sorted(data[data[group_on].isin(items[train_ix].tolist())]['ImageId'].tolist())
    val_ids = sorted(data[data[group_on].isin(items[val_ix].tolist())]['ImageId'].tolist())

    return train_ids, val_ids


def chunkify(lst, chunk_size=10, stride=None):
    stride = stride or chunk_size
    for x in range(0, len(lst), stride):
        yield lst[x:x + chunk_size]


def plot_single_scale(scale_lst, size=22, labels=None, fontsize=25):
    from matplotlib import cm as cm, pylab as plt
    plt.rcParams['figure.figsize'] = size, size // 2
    plt.figure()
    for i in list(range(0, len(scale_lst))):
        s = plt.subplot(1, len(scale_lst), i + 1)
        plt.imshow(scale_lst[i], cmap=cm.Greys_r)
        s.set_xticklabels([])
        s.set_yticklabels([])
        s.yaxis.set_ticks_position('none')
        s.xaxis.set_ticks_position('none')
        if labels and len(labels) > i:
            plt.title(labels[i], fontsize=fontsize)
    plt.tight_layout()


def flatten(lis):
    """Given a list, possibly nested to any level, return it flattened."""
    new_lis = []
    for item in lis:
        if not isinstance(item, six.string_types):
            new_lis.extend(flatten(item))
        else:
            new_lis.append(item)
    return new_lis


def get_unique_stats(stats, sort='loss', ascending=False):
    return stats.reset_index().sort_values(sort, ascending=ascending).groupby(
        ['experiment', 'job']).first().reset_index().set_index(['name']).sort_values(sort, ascending=ascending)


def best_models(stats, metric='loss', N=4, ascending=False, min_value=None, max_value=None,
                name_col='max_epoch_name', split_seeds=None, val_sizes=None):
    stats_unique = get_unique_stats(stats, metric, ascending).copy().reset_index()

    if min_value is not None:
        stats_unique = stats_unique[stats_unique[metric] > min_value]

    if max_value is not None:
        stats_unique = stats_unique[stats_unique[metric] < max_value]

    assert len(stats_unique)

    if val_sizes is None:
        val_sizes = stats_unique['data.val_size'].unique().tolist()

    if split_seeds is None:
        split_seeds = stats_unique['data.random_split_state'].unique().tolist()

    models = []
    for vs in val_sizes:
        for s in split_seeds:
            st = stats_unique[(stats_unique['data.val_size'].apply(str) == str(vs)) &
                              (stats_unique['data.random_split_state'].apply(str) == str(s))
                              ]

            m = st.iloc[:N][name_col].unique().tolist()

            print("[{}]".format(str((str(vs), str(s)))), len(m))
            if len(m):
                models.extend(m)

    return models

import inspect
import logging

import numpy as np
import six
from torch.optim import lr_scheduler

logger = logging.getLogger(__name__)


def get_default_args(func):
    """
    Returns a dictionary of arg_name:default_values for the input function
    """
    if six.PY3:
        spec = inspect.getfullargspec(func)
        defaults = dict(zip(spec.args[::-1], (spec.defaults or ())[::-1]))
        defaults.update(spec.kwonlydefaults or {})
        return defaults

    if six.PY2:
        args, varargs, keywords, defaults = inspect.getargspec(func)
        if defaults:
            return dict(zip(args[-len(defaults):], defaults))
        return {}
    raise RuntimeError('Not PY3 or PY2?... six might be broken!')


class PolyScheduler(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, max_epoch=100, exp=0.9):
        self.max_epoch = max_epoch
        self.exp = exp
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    def get_lr(self):
        mult = pow((1 - 1.0 * self.last_epoch / self.max_epoch), self.exp)
        return [group['lr'] * mult
                for group in self.optimizer.param_groups]


class MultiTypeLRScheduler(lr_scheduler._LRScheduler):

    def __init__(self, optimizer, last_epoch=-1, schedulers=None, change_epochs=None):

        if schedulers is None:
            schedulers = []

        if change_epochs is None:
            change_epochs = []

        assert len(change_epochs) + 1 >= len(schedulers), "change_epoch length must be at least len(schedulers)-1"

        self.schedulers = schedulers
        self.change_epochs = np.array(change_epochs)
        self._scheduler = self.schedulers[0]
        self.last_epoch = 0
        super().__init__(optimizer=optimizer, last_epoch=last_epoch)

    @property
    def scheduler(self):
        return self._scheduler

    @scheduler.setter
    def scheduler(self, sch):
        if sch != self._scheduler:
            logger.info("current epoch: %d, switching to %s.", self.last_epoch, str(sch))
            sch.base_lrs = self.base_lrs
            self._scheduler = sch

    @property
    def base_lrs(self):
        return self.scheduler.base_lrs

    @base_lrs.setter
    def base_lrs(self, values):
        self.scheduler.base_lrs = values

    def _check_scheduler(self):
        x = self.change_epochs > self.last_epoch
        self.scheduler = self.schedulers[np.argmax(x) if x.sum() else -1]

    def step(self, epoch=None):
        self._check_scheduler()
        self.scheduler.step(epoch)
        super().step(epoch)

    def get_lr(self):
        self._check_scheduler()
        return self.scheduler.get_lr()


class WarmupLRScheduler(lr_scheduler._LRScheduler):
    proxy_types = {
        'step': lr_scheduler.StepLR,
        'cos': lr_scheduler.CosineAnnealingLR,
        'multistep': lr_scheduler.MultiStepLR,
        'lambda': lr_scheduler.LambdaLR,
        'exp': lr_scheduler.ExponentialLR,
        # 'plateau': lr_scheduler.ReduceLROnPlateau,
        'cyclic': lr_scheduler.CyclicLR,
        'cos-restarts': lr_scheduler.CosineAnnealingWarmRestarts,
        'poly': PolyScheduler,
    }

    def __init__(self, optimizer, last_epoch=-1, main_scheduler=None, mode=None, warmup_epochs=1, multiplier=2.0,
                 **proxy_args):

        if mode is None or mode.lower() == "none":
            mode = None
        else:
            mode = mode.lower()
            assert mode in self.proxy_types, '{} is not a correct type'.format(mode)

        if mode is not None and main_scheduler is not None:
            raise ValueError('passed both main and a scheduler')

        if mode is not None and main_scheduler is None:
            main_scheduler = self.proxy_types[mode](optimizer, last_epoch=last_epoch, **proxy_args)

        self.main_scheduler = main_scheduler

        self.mode = mode
        self.warmup_epochs = warmup_epochs
        self.multiplier = multiplier
        if self.multiplier < 1.:
            raise ValueError('multiplier should be greater than 1.')

        self.warmed = False

        if self.warmup_epochs == 0:
            self.warmed = True
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self.last_epoch > self.warmup_epochs:
            if self.main_scheduler:
                if not self.warmed:
                    self.main_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.warmed = True
                return self.main_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.warmup_epochs + 1.) for base_lr in
                self.base_lrs]

    def step(self, epoch=None):
        if self.warmed and self.main_scheduler:
            if epoch is None:
                self.main_scheduler.step(epoch)
            else:
                self.main_scheduler.step(epoch - self.warmup_epochs)
        super().step(epoch)


proxy_types = {
    'step': lr_scheduler.StepLR,
    'cos': lr_scheduler.CosineAnnealingLR,
    'multistep': lr_scheduler.MultiStepLR,
    'lambda': lr_scheduler.LambdaLR,
    'exp': lr_scheduler.ExponentialLR,
    'cyclic': lr_scheduler.CyclicLR,
    'cos-restarts': lr_scheduler.CosineAnnealingWarmRestarts,
    'poly': PolyScheduler,
    'warm': WarmupLRScheduler,
}


def get_scheduler(optimizer, schedulers, steps_per_epoch=1,
                  change_epochs=None, scale_params=False,
                  mult_scale_params=None,
                  div_scale_params=None):
    mult_scale_params = mult_scale_params or ['max_epoch', 'step_size', 'milestones', 'T_max', 'patience', 'cooldown',
                                              'step_size_up', 'step_size_down', 'T_0', ]
    div_scale_params = div_scale_params or ['exp']

    def _convert_value_mult(v, steps_per_epoch):
        if isinstance(v, list):
            return [_convert_value_mult(vv, steps_per_epoch) for vv in v]
        return v * steps_per_epoch

    def _convert_value_div(v, steps_per_epoch):
        if isinstance(v, list):
            return [_convert_value_div(vv, steps_per_epoch) for vv in v]
        return v / steps_per_epoch

    def _convert_params(p, steps_per_epoch=1, scale_params=True):
        out = {}
        for k, v in p.items():
            if scale_params:
                if k in mult_scale_params:
                    v = _convert_value_mult(v, steps_per_epoch)
                elif k in div_scale_params:
                    v = _convert_value_div(v, steps_per_epoch)
            out[k] = v
        return out

    change_epochs = change_epochs or []
    change_epochs = _convert_value_mult(change_epochs, steps_per_epoch=steps_per_epoch)

    if isinstance(schedulers, dict):
        schedulers = [schedulers]

    schedulers_ = []
    for p in schedulers:
        klass = proxy_types[p.pop('type', 'warm')]
        params = get_default_args(klass)
        params.update(p)
        logger.info('params before: %s', str(params))
        params = _convert_params(params, steps_per_epoch, scale_params=scale_params)
        logger.info('params after scaling: %s', str(params))
        schedulers_.append(klass(optimizer, **params))  # warm default is for backwards_compat

    print(schedulers_)
    scheduler = MultiTypeLRScheduler(optimizer, schedulers=schedulers_, change_epochs=change_epochs)
    return scheduler

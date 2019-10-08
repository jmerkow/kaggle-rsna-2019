from collections import defaultdict

import torch
from torch import nn


class FocalLoss(nn.Module):
    loss_func = nn.BCELoss

    def __init__(self, alpha=1, gamma=2, reduction='mean', **kwargs):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = self.loss_func(reduction='none', **kwargs)

    def forward(self, inputs, targets):
        BCE_loss = self.bce_loss(inputs, targets)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        elif self.reduction == 'none':
            return F_loss

        raise NotImplementedError('bad reduction type')


class FocalWithLogitsLoss(FocalLoss):
    loss_func = nn.BCEWithLogitsLoss


class Criterion(object):

    def __init__(self, classes=None, tasks=None, **defaults):

        self.classes = classes
        # self.defaults = defaults
        self.default_criterion = _Criterion(**defaults).cuda()
        self.criteria = defaultdict(lambda: _Criterion(**defaults).cuda())

        tasks = tasks or {}
        for task in tasks:
            params = defaults.copy()
            params.update(tasks.get(task, {}))
            self.criteria[task] = _Criterion(**params).cuda()

    def __call__(self, scores, targets):

        self.losses = []
        metrics = {}

        if isinstance(scores, dict):
            for task, sc in scores.items():
                criterion = self.criteria[task]
                loss = criterion(sc, targets)
                metrics['{}/loss'.format(task)] = loss.cpu().detach().numpy()
                metrics.update({'{}/loss-{}'.format(task, cl_name): l.cpu().detach().numpy()
                                for cl_name, l in zip(self.classes, criterion.raw_loss.T)})
                self.losses.append(loss)
        else:
            criterion = self.default_criterion
            loss = criterion(scores, targets)
            metrics['loss'] = loss.cpu().detach().numpy()
            metrics.update({'loss-{}'.format(cl_name): l.cpu().detach().numpy()
                            for cl_name, l in zip(self.classes, criterion.raw_loss)})

        return metrics

    def backward(self):
        for loss in self.losses:
            loss.backward(retain_graph=True)


class _Criterion(nn.Module):
    loss_types = [
        {
            'bce': torch.nn.BCELoss,
            'ce': torch.nn.NLLLoss,
            'focal': FocalLoss,
        },
        {
            'bce': torch.nn.BCEWithLogitsLoss,
            'ce': torch.nn.CrossEntropyLoss,
            'focal': FocalWithLogitsLoss
        }]

    def get_loss(self, type, logits=True, **loss_params):
        klass = self.loss_types[int(logits)][type]
        if 'pos_weight' in loss_params:
            if not logits or type == 'ce':
                raise NotImplementedError('pos_weight not implemented for this loss')
            pos_weight = loss_params['pos_weight']
            if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight).float()
                loss_params['pos_weight'] = pos_weight
        loss_params['reduction'] = 'none'
        return klass(**loss_params)

    def __init__(self, loss_weights=None, reduction='mean', type='bce', logits=True, **loss_params):
        super().__init__()
        assert reduction in ['none', 'sum', 'mean'], "bad reduction type"
        assert type in self.loss_types[logits], "bad loss type"

        if loss_weights is not None and not isinstance(loss_weights, torch.Tensor):
            loss_weights = torch.tensor(loss_weights, requires_grad=False).float()

        self.raw_loss = None
        self.reduction = reduction
        self.loss_weights = loss_weights
        self.register_buffer('loss_weights_const', self.loss_weights)
        self.criteria = self.get_loss(type=type, logits=logits, **loss_params)

    def __call__(self, scores, targets):

        batch_size = len(scores)
        raw_loss = self.criteria(scores, targets)
        if self.loss_weights is not None:
            weighted_loss = raw_loss * self.loss_weights_const
        else:
            weighted_loss = raw_loss

        if self.reduction == 'none':
            loss = weighted_loss.sum(dim=1)
            self.raw_loss = raw_loss.detach()
        else:
            loss = weighted_loss.sum()
            self.raw_loss = raw_loss.sum(dim=1).detach()

        if self.reduction == 'mean':
            loss /= batch_size

        return loss

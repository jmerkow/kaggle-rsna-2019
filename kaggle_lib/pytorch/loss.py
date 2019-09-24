import torch
from torch import nn

score_loss_types = {'bce': torch.nn.BCEWithLogitsLoss}


class Criterion(nn.Module):
    loss_types = {
        'bce': torch.nn.BCEWithLogitsLoss,
        'ce': torch.nn.CrossEntropyLoss,
    }

    def get_loss(self, type, *args, **loss_params):
        klass = self.loss_types[type]
        if 'pos_weight' in loss_params:
            pos_weight = loss_params['pos_weight']

            if pos_weight is not None and not isinstance(pos_weight, torch.Tensor):
                pos_weight = torch.tensor(pos_weight).float()
                loss_params['pos_weight'] = pos_weight
        return klass(**loss_params)

    def __init__(self, loss_params, loss_reduction='mean', **defaults):
        super().__init__()
        self.raw_loss = None

        real_params = {}
        self.targets = {}
        self.loss_weights = {}
        for name, lp in loss_params.items():
            temp = defaults.copy()
            temp.update(lp)
            self.targets[name] = temp.pop('target', name)
            self.loss_weights[name] = temp.pop('loss_weight', 1)
            real_params[name] = temp

        if loss_reduction == 'mean':
            norm = sum(self.loss_weights.values())
            self.loss_weights = {name: lw / norm for name, lw in self.loss_weights.items()}
        self.losses = nn.ModuleDict({name: self.get_loss(**lp) for name, lp in real_params.items()})

    def forward(self, output_dict, targets):
        self.raw_loss = {}  # so we can capture it if we want
        loss = 0
        for name, output in output_dict.items():
            if name not in self.losses:
                continue
            target = targets[self.targets[name]]
            lw = self.loss_weights[name]
            temp = self.losses[name](output, target)
            self.raw_loss[name] = temp.detach()
            loss += lw * temp
        return loss

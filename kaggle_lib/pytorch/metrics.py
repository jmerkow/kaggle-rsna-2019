import pandas as pd

from .loss import Criterion


def get_loss_label(df, pred_col, label_col):
    pos_gt = df.loc[df[label_col] == 1.0]
    neg_gt = df.loc[df[label_col] == 0.0]

    pos_loss = df.loc[pos_gt.index][pred_col].mean()
    neg_loss = df.loc[neg_gt.index][pred_col].mean()

    return pos_loss, neg_loss


class RSNA2019Metric(object):

    def __init__(self, classes, image_id_key='image_id', **kwargs):
        kwargs['reduction'] = 'none'
        self.classes = classes
        self.image_id_key = image_id_key
        self.criteria = Criterion(classes=classes, **kwargs)

    def add_batch(self, scores, target, sample):

        image_ids = sample.get(self.image_id_key)
        metrics = self.criteria(scores, target)
        if isinstance(scores, dict):
            scores = {k: v.cpu().detach().numpy() for k, v in scores.items()}
        else:
            scores = scores.cpu().detach().numpy()
        for i, ImageId in enumerate(image_ids):
            row = {
                'ImageId': ImageId,
            }

            row.update({name: m[i] for name, m in metrics.items()})
            if isinstance(scores, dict):
                row.update({'{}/score_{}'.format(name, c): r
                            for name, scoress in scores.items() for c, r in zip(self.classes, scoress[i])})
            else:
                row.update({'score_' + c: r for c, r in zip(self.classes, scores[i])})
            row.update({'label_' + c: r for c, r in zip(self.classes, target[i])})

            self.rows.append(row)

    def reset(self):
        self.rows = []

    def get_rows(self):
        return self.rows

    def get_df(self):
        df = pd.DataFrame(self.get_rows()).drop_duplicates()
        return df

    @property
    def mean(self):
        df = self.get_df()
        loss_cols = [c for c in list(df) if c.startswith('loss')]

        temp = {cl: get_loss_label(df, pred_col='loss-{}'.format(cl),
                                   label_col='label_{}'.format(cl))
                for cl in self.classes}

        pos_losses = {'pos-loss-{}'.format(cl): v[0] for cl, v in temp.items()}
        neg_losses = {'neg-loss-{}'.format(cl): v[1] for cl, v in temp.items()}

        losses = df[loss_cols].mean().to_dict()

        losses.update(pos_losses)
        losses.update(neg_losses)

        return losses

# class BalancedBinaryAccuracy(nn.Module):
#     __name__ = 'balanced_acc'
#
#     def __init__(self, threshold=.5, activation='sigmoid'):
#         super().__init__()
#         self.threshold = threshold
#         self.activation = activation
#
#     def forward(self, y_pr, y_gt):
#         with torch.no_grad():
#             tp, tn, fp, fn = binary_counts(y_pr, y_gt, threshold=self.threshold, activation=self.activation)
#             p = tp + fn
#             n = tn + fp
#             tpr = (tp / p) if p > 0 else torch.ones_like(p).float()
#             tnr = (tn / n) if n > 0 else torch.ones_like(n).float()
#             x = (tpr + tnr) / 2.
#         return x
#
#
# class BinaryAccuracy(nn.Module):
#     __name__ = 'acc'
#
#     def __init__(self, threshold=.5, activation='sigmoid'):
#         super().__init__()
#         self.threshold = threshold
#         self.activation = activation
#
#     def forward(self, y_pr, y_gt):
#         tp, tn, fp, fn = binary_counts(y_pr, y_gt, threshold=self.threshold, activation=self.activation)
#         return (tp + tn) / (tp + tn + fp + fn)

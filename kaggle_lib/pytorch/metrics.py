import pandas as pd

from .loss import Criterion


class RSNA2019Metric(object):
    image_id_key = 'image_id'

    def __init__(self, class_order, **kwargs):
        kwargs['reduction'] = 'none'
        self.class_order = class_order
        self.criteria = Criterion(**kwargs).cuda()

    def add_batch(self, scores, target, sample):

        image_ids = sample.get(self.image_id_key)
        target = target
        batch_size = len(scores)

        loss = self.criteria(scores, target).cpu().detach().numpy()
        raw_loss = self.criteria.raw_loss.cpu().detach().numpy()
        scores = scores.cpu().detach().numpy()

        if batch_size == 1:
            loss = [loss]
            raw_loss = [raw_loss]
            image_ids = [image_ids]

        for i, ImageId in enumerate(image_ids):
            row = {
                'ImageId': ImageId,
                'loss': loss[i]
            }
            row.update({'loss_' + c: r for c, r in zip(self.class_order, raw_loss[i])})
            row.update({'score_' + c: r for c, r in zip(self.class_order, scores[i])})
            self.rows.append(row)

        del self.criteria.raw_loss

    def reset(self):
        self.rows = []

    def get_rows(self):
        return self.rows

    def get_df(self):
        return pd.DataFrame(self.get_rows()).drop_duplicates()

    @property
    def mean(self):
        df = self.get_df()
        loss_cols = [c for c in list(df) if c.startswith('loss')]
        return df[loss_cols].mean().to_dict()

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

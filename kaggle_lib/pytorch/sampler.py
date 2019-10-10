from torch.utils.data.sampler import Sampler
import numpy as np


class LabelSampler(Sampler):
    def __init__(self, ids, data, label, start_rate = 1.0, end_rate=1.0, cycle_length=5.0, last_epoch=-1):
        idxs = np.array(list(ids.keys()))
        labels = np.array([data[img_id]['label__' + label] for img_id in ids.values()])
        self.pos_idxs = idxs[np.nonzero(labels)]
        self.neg_idxs = idxs[np.nonzero(labels == 0)]

        self.start_rate = start_rate
        self.end_rate = end_rate
        self.cycle_length = cycle_length
        self.n_positive = sum(labels)

        self.last_epoch = last_epoch
        self.step()

    def __iter__(self):
        negative_sample = np.random.choice(self.neg_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.pos_idxs)))
        return iter(shuffled.tolist())

    def step(self, epoch=None):
        if epoch is None:
            self.last_epoch += 1
        else:
            self.last_epoch = epoch
        cycle_epoch = self.last_epoch % self.cycle_length
        pos_freq = self.start_rate + ((self.end_rate - self.start_rate)/self.cycle_length) * cycle_epoch
        self.update_pos_freq(pos_freq)

    def update_pos_freq(self, pos_freq):
        self.pos_freq = pos_freq
        self.n_negative = int(self.n_positive * (1 - self.pos_freq) / self.pos_freq)

    def __len__(self):
        return self.n_positive + self.n_negative

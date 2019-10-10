from torch.utils.data.sampler import Sampler
import numpy as np


class LabelSampler(Sampler):
    def __init__(self, ids, data, label, cur_epoch=0, start_pos_freq = 1.0, end_pos_freq=1.0, cycle_length=5.0):
        idxs = np.array(list(ids.keys()))
        labels = np.array([data[img_id]['label__' + label] for img_id in ids.values()])
        self.pos_idxs = idxs[np.nonzero(labels)]
        self.neg_idxs = idxs[np.nonzero(labels == 0)]

        self.start_pos_freq = start_pos_freq
        self.end_pos_freq = end_pos_freq
        self.cycle_length = cycle_length
        self.n_positive = sum(labels)

        self.step(cur_epoch)

    def __iter__(self):
        negative_sample = np.random.choice(self.neg_idxs, size=self.n_negative)
        shuffled = np.random.permutation(np.hstack((negative_sample, self.pos_idxs)))
        return iter(shuffled.tolist())

    def step(self, epoch_dec):
        cycle_epoch_dec = epoch_dec % self.cycle_length
        pos_freq = self.start_pos_freq + ((self.end_pos_freq - self.start_pos_freq)/self.cycle_length) * cycle_epoch_dec
        self.update_pos_freq(pos_freq)

    def update_pos_freq(self, pos_freq):
        self.pos_freq = pos_freq
        self.n_negative = int(self.n_positive * (1 - self.pos_freq) / self.pos_freq)

    def __len__(self):
        return self.n_positive + self.n_negative


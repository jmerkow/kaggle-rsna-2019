from __future__ import absolute_import, print_function

from joblib import Parallel, delayed
from torch.utils.data._utils.collate import default_collate
import random
import pandas as pd


def get_ds_data(rows, dataset):
    output = []
    for row in rows:
        output.append(dataset[row])
    return output


class CustomDataLoader(object):

    def __init__(self, dataset, batch_size = 1, shuffle = False, num_workers = None, backend='loky',
                 img_ids=None,
                 *args,
                 **filter_params):
        super(CustomDataLoader, self).__init__()
        self.dataset = dataset
        # keep data in dataloader object to reduce size of memory copies
        data = pd.read_csv(self.dataset.get_csv_file()).set_index('ImageId')
        img_ids = img_ids or data.index.tolist()
        random.shuffle(img_ids)
        img_ids = self.apply_filter(img_ids, **filter_params)
        data = data.loc[img_ids]
        self.data = data.reset_index()

        self.shuffle = shuffle
        self.batcher = list(range(len(self.data)))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.chunk_size = self.batch_size // self.num_workers
        self.backend = backend
        self.reset()

    def apply_filter(self, img_ids, **filter_params):
        # place holder for now
        return img_ids

    def __iter__(self):
        return self

    def __next__(self):
        """
        """
        if (self.curr_i * self.batch_size) >= len(self.data):
            self.reset()
            raise StopIteration
        else:
            batch_indices = self.batcher[self.curr_i * self.batch_size:(self.curr_i + 1) * self.batch_size]
            batch_chunks = [batch_indices[j*self.chunk_size:(j+1)*self.chunk_size] for j in range(self.num_workers)]
            batch_chunks_rows = [[self.data.loc[index] for index in chunk] for chunk in batch_chunks]
            batch_data = Parallel(n_jobs=self.num_workers, backend=self.backend)(delayed(get_ds_data)(batch_chunk_rows,
                                                                                                      self.dataset)
                                                                                 for batch_chunk_rows in batch_chunks_rows)
            self.curr_i = self.curr_i + 1
            batch_data = [item for chunk in batch_data for item in chunk]
            batch_data = default_collate(batch_data)
            return batch_data

    def reset(self):
        self.curr_i = 0
        if self.shuffle:
            random.shuffle(self.batcher)

    def __len__(self):
        return len(self.data) // self.batch_size
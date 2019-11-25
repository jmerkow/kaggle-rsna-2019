import pandas as pd
import torch

from kaggle_lib.pytorch.ensembler import Ensembler
from kaggle_lib.pytorch.submission import median_filter_submissions

submission_headers_csv = '/data/rsna-intracranial-hemorrhage-detection/stage1-test-headers.csv'
model_root = '/mnt/nas/experiments-kaggle/rsna2019/'
dataset = 'rsna2019-stage1'
sub_len = 471270

k = 10
best_file_loc = "/mnt/nas/experiments-kaggle/rsna2019/amvepa/best_metrics_rwt_v1.csv"
best_stats = pd.read_csv(best_file_loc)

filt_best_stats = best_stats.loc[best_stats["data.random_split_group"] == 'patient_id']
filt_best_stats = filt_best_stats.loc[filt_best_stats["data.n_splits"] == 10]

filt_best_stats = filt_best_stats.loc[(filt_best_stats["model.encoder"] == "inceptionv4") |
                                      (filt_best_stats["model.encoder"] == "se_resnext50_32x4d")]
fold0_best_stats = filt_best_stats.loc[filt_best_stats["data.fold"] == 0]
fold1_best_stats = filt_best_stats.loc[filt_best_stats["data.fold"] == 1]

fold0_best_stats = fold0_best_stats.sort_values(by=["rwt-loss"])
models = list(fold0_best_stats["name"][:k])

fold1_best_stats = fold1_best_stats.sort_values(by=["rwt-loss"])
models_ = list(fold1_best_stats["name"][:k])

models.extend(models_)
print(models)
print(len(models))

# If you leave out output_dir it will write to the model_root, which is fine.
e = Ensembler(model_root=model_root, output_dir=None, dataset=dataset)
print(e)

with torch.cuda.device(1):  # whatever device you want
    csv_file = e.ensemble(models, random_count=4)

print(csv_file)
submission = median_filter_submissions(csv_file, submission_headers_csv, r=5, avg=0.7, sub_len=sub_len)

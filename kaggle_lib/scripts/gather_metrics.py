import glob
import os

import pandas as pd

from kaggle_lib.pytorch.result_collection import batch_metrics_to_json, merge_json_metrics, generate_rwt_losss_csv
from kaggle_lib.pytorch.trainers import ClassifierTrainer

metrics_glob = '*/*/*/metrics/*.json'
output_filename = 'best_metrics.csv'
output_filename2 = 'best_metrics_rwt.csv'

checkpoints = glob.glob('/mnt/nas/experiments-kaggle/rsna2019/*/*/*/checkpoints/model_step002.pth.tar')
checkpoints += glob.glob('/mnt/nas/experiments-kaggle/rsna2019/*/*/*/checkpoints/model_step004.pth.tar')
checkpoints += glob.glob('/mnt/nas/experiments-kaggle/rsna2019/*/*/*/checkpoints/model_step006.pth.tar')

checkpoints = list(set(checkpoints))

bad_checkpoints = [f for f in checkpoints if not os.path.exists(f)]
checkpoints = [f for f in checkpoints if os.path.exists(f)]

print(len(checkpoints), len(bad_checkpoints))

batch_metrics_to_json(checkpoints)

defaults = pd.io.json.json_normalize(ClassifierTrainer.section_defaults).T.to_dict()[0]

stats = merge_json_metrics('/mnt/nas/experiments-kaggle/rsna2019/',
                           metrics_glob=metrics_glob,
                           defaults=defaults,
                           output_filename=output_filename,

                           )

generate_rwt_losss_csv(stats, new_rwt_csv='/mnt/nas/experiments-kaggle/rsna2019/' + output_filename2)

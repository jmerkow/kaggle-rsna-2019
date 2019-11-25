import os

import pandas as pd
import tqdm

from kaggle_lib.pytorch.ensembler import make_submission_df


def make_model_name(model_name):
    bad_names = ['mnt', 'nas', 'experiments-kaggle', 'rsna2019', 'checkpoints', 'results', 'rsna2019-stage1_test']
    model_name = '_'.join([s for s in model_name.strip(os.sep, ).split(os.sep) if s not in bad_names]).replace(
        '.pth.tar', '')
    return '{}'.format(model_name)


def median_filter_submissions(csv_file_,
                              submission_headers_csv_='/data/rsna-intracranial-hemorrhage-detection/stage1-test-headers.csv',
                              avg=.7, r=5, func='median', sub_len=471270):
    scores_df = pd.read_csv(csv_file_).set_index('ID')
    submission_headers = pd.read_csv(submission_headers_csv_).set_index('ImageId')
    score_cols = list(scores_df)
    merged_df = scores_df.join(submission_headers)

    new_grps = []
    for _, grp in tqdm.tqdm(list(merged_df.groupby('series_instance_uid'))):
        grp = grp.set_index('ipp_z_norm').sort_index()

        lgrp = len(grp)
        new_grp = {}
        for c in score_cols:
            new_grp[c] = grp[c].rolling(r, center=True).median()
            new_grp[c] = new_grp[c].fillna(grp[c])
            if 1 - avg:
                new_grp[c] = (1 - avg) * grp[c] + avg * new_grp[c]
        new_grp = pd.concat(new_grp, axis=1)
        new_grp = grp.drop(score_cols, axis=1).join(new_grp)
        new_grps.append(new_grp)
    new_grps = pd.concat(new_grps)
    new_scores_df = new_grps.rename(columns={'sop_instance_uid': 'ID'}).set_index(['ID'])[score_cols].sort_index()

    new_csv_file = os.path.splitext(csv_file_)[0] + '-Rs{}{}Avg{}.csv'.format(r, func, avg)
    new_scores_df.to_csv(new_csv_file)
    output_csv = './submission_' + make_model_name(new_csv_file) + '_clip{}.csv'

    print(new_csv_file)
    clip = 0
    sub_df = make_submission_df(new_csv_file, clip=clip)
    print(len(sub_df))
    assert len(sub_df) == sub_len
    fn = output_csv.format(clip)
    print(fn)
    sub_df.to_csv(fn)
    return fn

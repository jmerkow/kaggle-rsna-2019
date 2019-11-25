import glob
import json
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
import yaml
from joblib import Parallel, delayed

from kaggle_lib.pytorch.datasets import get_data_constants
from kaggle_lib.pytorch.utils import chunkify


def metrics_to_json(checkpoint, force_overwrite=False, verbose=True):
    job_dir = os.path.dirname(os.path.dirname(checkpoint))
    train_yml = os.path.join(job_dir, 'train.yml')

    epoch_rex = r'model_step(\d*).pth.tar'

    chpk = None
    epoch = None

    match = re.search(epoch_rex, checkpoint)

    if match:
        epoch = int(match.groups()[0])
    else:
        try:
            if verbose:
                print('no match, loading model for epoch')
            chpk = torch.load(checkpoint, map_location='cpu')
        except (EOFError, RuntimeError):
            return -1
        epoch = chpk['epoch']

    assert epoch is not None, 'crash!'
    info = {
        'epoch': epoch,
        'experiment': os.path.basename(os.path.dirname(job_dir)),
        'exp_group': os.path.basename(os.path.dirname(os.path.dirname(job_dir))),
        'job': os.path.basename(job_dir),
    }

    with open(train_yml) as f:
        config = yaml.safe_load(f)['cfg']

    save_loc = 'metrics/model_step{:03d}.json'.format(epoch)
    save_loc = os.path.join(job_dir, save_loc)

    if verbose:
        print(save_loc)
    if os.path.exists(save_loc) and not force_overwrite:
        return 0 if chpk is None else -2

    if chpk is None:
        try:
            chpk = torch.load(checkpoint, map_location='cpu')
        except (EOFError, RuntimeError):
            return -1

    if not os.path.exists(os.path.dirname(save_loc)):
        os.makedirs(os.path.dirname(save_loc))
    ds = dict()
    metrics = chpk['val_metrics'].copy()
    info.update(pd.io.json.json_normalize(config).T.to_dict()[0])
    ds['metrics'] = metrics
    ds['info'] = info

    with open(save_loc, 'w') as f:
        json.dump(ds, f, indent=2)
    return 1


def batch_metrics_to_json(checkpoints, n_jobs=5, chunk_size=10, force_overwrite=False, verbose=False):
    eofs = 0
    skips = 0
    written = 0
    load = 0
    errors = []

    checkpoints = sorted(checkpoints, key=lambda x: os.path.getmtime(x), reverse=True)

    chunks = list(chunkify(checkpoints, chunk_size=chunk_size))
    pbar = tqdm.tqdm(total=len(checkpoints), desc='overall', )

    with Parallel(n_jobs=n_jobs, prefer='threads') as parallel:
        for chunk in chunks:
            pbar.set_postfix_str('errors {}, e:{},s:{},w:{},l:{}'.format(len(errors), eofs, skips, written, load))
            try:
                results = parallel(
                    delayed(metrics_to_json)(fn, force_overwrite=force_overwrite, verbose=verbose) for fn in chunk)
                eofs += sum([r == -1 for r in results])
                skips += sum([r == 0 or r == -2 for r in results])
                load += sum([r == -2 for r in results])
                written += sum([r == 1 for r in results])

            except KeyboardInterrupt:
                raise
            except BaseException as e:
                errors.append((chunk, str(e), type(e)))
            pbar.update(len(chunk))
    return errors


def get_log_modified(sub, top_dir):
    log_file = 'logs/*.out'
    fns = glob.glob(os.path.join(top_dir, sub, log_file))
    fns = sorted(fns, key=lambda x: os.path.getsize(x), reverse=True)
    fn = fns[0]
    return datetime.fromtimestamp(os.path.getmtime(fn))


def get_plateau_data(sub, top_dir):
    log_file = 'logs/*.out'
    fn = sorted(glob.glob(os.path.join(top_dir, sub, log_file)),
                key=lambda x: os.path.getsize(x), reverse=True)[0]
    re_reduce = "^Epoch\s+(?P<epoch>[\d]*):\s+reducing learning rate of group 0 to (?P<lr>[\d|\.|e|-]+).$"
    re_saves = 'saving to checkpoints/model_step(\d+).pth.tar'
    reduces = []
    saves = []

    with open(fn, 'r') as f:
        for line in f.readlines():
            match = re.search(re_reduce, line)
            if match:
                reduces.append((int(match.groupdict()['epoch']), float(match.groupdict()['lr'])))
            match = re.search(re_saves, line)
            if match:
                saves.append(int(match.groups()[0]))

    if len(reduces):
        steps, lrs = zip(*reduces)
        steps = str(steps)
        gamma = round(float(10. ** (np.log10(1e-6) - np.log10(1e-5))), 5)
    else:
        steps, lrs, gamma = None, None, None
    max_epochs = max(saves)

    return max_epochs, steps, gamma


def merge_json_metrics(top_dir, metrics_glob='*/*/*/metrics/*.json', output_filename='best_metrics.csv', defaults=None,
                       force_reread=False):
    defaults = defaults or {}
    output_file = os.path.join(top_dir, output_filename)

    ignore = []
    stats = None
    if os.path.exists(output_file) and not force_reread:
        stats = pd.read_csv(output_file).set_index('name')
        ignore = [os.path.join(top_dir, fn) for fn in stats['json_file'].unique()]

    json_files = glob.iglob(os.path.join(top_dir, metrics_glob))
    json_files = [f for f in json_files if f not in ignore]
    print("num files:", len(json_files))

    def finish(m, i, **defaults_):
        defaults_ = defaults_.copy()
        defaults_.update(pd.io.json.json_normalize(i).T.to_dict()[0])
        return pd.concat([pd.io.json.json_normalize(m),
                          pd.io.json.json_normalize(defaults_)], axis=1).set_index('name')

    def load_row(fn):
        with open(fn, 'r') as f:
            data = json.load(f)
        info = data['info']
        info['job_dir'] = '{exp_group}/{experiment}/{job}'.format(**info)
        info['name'] = name = '{exp_group}/{experiment}/{job}/checkpoints/model_step{epoch:>03}.pth.tar'.format(**info)
        info['json_file'] = name = '{exp_group}/{experiment}/{job}/metrics/model_step{epoch:>03d}.json'.format(**info)
        # info['max_epochs'], info['steps'], info['gamma'] = get_plateau_data(info['job_dir'], top_dir=top_dir)
        info['last_mod'] = get_log_modified(info['job_dir'], top_dir=top_dir)
        row = (data['metrics'], info)
        return row

    rows = Parallel(n_jobs=10)(delayed(load_row)(fn) for fn in tqdm.tqdm(json_files))
    if len(rows):
        new_stats = pd.concat([finish(m, i, **defaults) for m, i in rows], sort=True)
        if stats is not None:
            stats = pd.concat([stats, new_stats], axis=0, sort=True)
        else:
            stats = new_stats.copy()
    stats.to_csv(output_file)
    return stats


def get_train_df(dataset='rsna2019-stage1', data_root='/data/', transforms=None,
                 model_preprocessing=None):
    datacatalog, dataset_map = get_data_constants(data_root)
    train_dataset_name = dataset_map[dataset]['train']
    train_catalog = datacatalog[train_dataset_name]
    csv_file = os.path.join(data_root, train_catalog['csv_file'])
    df = pd.read_csv(csv_file).set_index('ImageId')
    gt_cols = [c for c in list(df) if c.startswith('label')]
    return df[gt_cols]


def get_rwt_loss(pred_df, gt_df, gt_df_val, gt_label, pred_label):
    pos_prop_train = gt_df[gt_label].mean()

    pos_gt = gt_df_val.loc[gt_df_val[gt_label] == 1.0]
    neg_gt = gt_df_val.loc[gt_df_val[gt_label] == 0.0]

    pos_loss = pred_df.loc[pos_gt.index][pred_label].mean()
    neg_loss = pred_df.loc[neg_gt.index][pred_label].mean()

    rwt_loss = pos_loss * pos_prop_train + neg_loss * (1 - pos_prop_train)

    return rwt_loss


def get_rwt_val(gt_df, model_checkpoint):
    checkpoint = torch.load(model_checkpoint)
    pred_df = pd.DataFrame(checkpoint['scorecard'])
    pred_df = pred_df.set_index('ImageId')
    val_set = pred_df.index
    gt_df_val = gt_df.loc[val_set]

    # sdh
    sdh_loss = pred_df['loss-sdh'].mean()
    sdh_rwt_loss = get_rwt_loss(pred_df, gt_df, gt_df_val, 'label__subdural', 'loss-sdh')

    # sah
    sah_loss = pred_df['loss-sah'].mean()
    sah_rwt_loss = get_rwt_loss(pred_df, gt_df, gt_df_val, 'label__subarachnoid', 'loss-sah')

    # ivh
    ivh_loss = pred_df['loss-ivh'].mean()
    ivh_rwt_loss = get_rwt_loss(pred_df, gt_df, gt_df_val, 'label__intraventricular', 'loss-ivh')

    # iph
    iph_loss = pred_df['loss-iph'].mean()
    iph_rwt_loss = get_rwt_loss(pred_df, gt_df, gt_df_val, 'label__intraparenchymal', 'loss-iph')

    # edh
    edh_loss = pred_df['loss-edh'].mean()
    edh_rwt_loss = get_rwt_loss(pred_df, gt_df, gt_df_val, 'label__epidural', 'loss-edh')

    # any
    any_loss = pred_df['loss-any'].mean()
    any_rwt_loss = get_rwt_loss(pred_df, gt_df, gt_df_val, 'label__any', 'loss-any')

    loss = (1 / 7) * (sdh_loss + sah_loss + ivh_loss + iph_loss +
                      edh_loss) + (2 / 7) * any_loss
    rwt_loss = (1 / 7) * (sdh_rwt_loss + sah_rwt_loss + ivh_rwt_loss + iph_rwt_loss
                          + edh_rwt_loss) + (2 / 7) * any_rwt_loss

    return {"rwt-loss-sdh": sdh_rwt_loss,
            "rwt-loss-sah": sah_rwt_loss,
            "rwt-loss-ivh": ivh_rwt_loss,
            "rwt-loss-iph": iph_rwt_loss,
            "rwt-loss-edh": edh_rwt_loss,
            "rwt-loss-any": any_rwt_loss,
            "rwt-loss": rwt_loss,
            "loss-sdh": sdh_loss,
            "loss-sah": sah_loss,
            "loss-ivh": ivh_loss,
            "loss-iph": iph_loss,
            "loss-edh": edh_loss,
            "loss-any": any_loss,
            "loss": loss}


def generate_rwt_losss_csv(stats,
                           old_rwt_csv=None,
                           new_rwt_csv="/mnt/nas/experiments-kaggle/rsna2019/amvepa/best_metrics_rwt_v1.csv"):
    gt_df = get_train_df()
    if old_rwt_csv is not None:
        old_rwt_stats = pd.read_csv(old_rwt_csv).set_index('name')
        stats = stats.set_index('name')
        print(old_rwt_stats.index)
        # print(best_stats.index)
        stats[old_rwt_stats.index]["rwt-loss-sdh"] = old_rwt_stats["rwt-loss-sdh"]
        stats[old_rwt_stats.index]["rwt-loss-sah"] = old_rwt_stats["rwt-loss-sah"]
        stats[old_rwt_stats.index]["rwt-loss-ivh"] = old_rwt_stats["rwt-loss-ivh"]
        stats[old_rwt_stats.index]["rwt-loss-iph"] = old_rwt_stats["rwt-loss-iph"]
        stats[old_rwt_stats.index]["rwt-loss-edh"] = old_rwt_stats["rwt-loss-edh"]
        stats[old_rwt_stats.index]["rwt-loss-any"] = old_rwt_stats["rwt-loss"]
        stats[old_rwt_stats.index]["rwt-loss"] = old_rwt_stats["rwt-loss"]
        done_models = set(list(old_rwt_stats["name"]))
    else:
        done_models = set()

    model_paths = stats["name"]
    with torch.cuda.device(2):
        for model_path in model_paths:
            abs_model_path = os.path.join("/mnt/nas/experiments-kaggle/rsna2019", model_path)
            if model_path in done_models:
                print("Already completed: {}".format(abs_model_path))
                continue
            else:
                print(abs_model_path)
                try:
                    losses = get_rwt_val(gt_df, abs_model_path)
                    print(losses)
                except:
                    print('errored')
                    continue
                for k, v in losses.items():
                    if 'rwt' in k:
                        stats.loc[stats["name"] == model_path, k] = v
    stats.to_csv(new_rwt_csv, index=False)
    return stats

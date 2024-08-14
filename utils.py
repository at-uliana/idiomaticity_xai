import pandas as pd
import os
import json


def is_model_on_gpu(model):
    return next(model.parameters()).is_cuda


def are_all_model_parameters_on_gpu(model):
    return all(p.is_cuda for p in model.parameters())


def are_all_model_buffers_on_gpu(model):
    return all(b.is_cuda for b in model.buffers())


def make_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def save_experiment_data(experiment_data, outdir):
    if os.path.exists(outdir):
        print("Previous experiment detected.")
        print("Added new data to the previous experiment.")
        previous_experiment = json.load(open(outdir, 'r'))
        for data in experiment_data:
            previous_experiment.append(data)
        json.dump(previous_experiment, open(outdir, 'w'), indent=True)
    else:
        json.dump(experiment_data, open(outdir, 'w'), indent=True)


def train_test_dev_split(data_path, split_path):
    cols = ['idiom', 'sentence', 'paragraph', 'label', 'split']
    data = pd.read_csv(data_path, sep='\t')
    split = pd.read_csv(split_path, sep='\t')
    data['split'] = split['split']
    train_data = data[data['split'] == 'train']
    test_data = data[data['split'] == 'test']
    dev_data = data[data['split'] == 'dev']
    return train_data[cols], dev_data[cols], test_data[cols]

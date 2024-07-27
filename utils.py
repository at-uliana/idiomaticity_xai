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


class ExperimentConfig:
    # Temporary class to store configurations
    # (to be replaced with `sacred` library or similar)
    def __init__(self, config_file):
        config = json.load(open(config_file, 'r'))
        self.seed = config['seed']
        self.learning_rate = config['learning rate']
        self.batch_size = config['batch size']
        self.max_length = config['max length']
        self.n_epochs = config['n epochs']
        self.data_file = config['data file']
        self.split_file = config['split file']
        self.model_name = config['model name']
        self.output_dir = config['output dir']
        self.freeze = config['freeze']
        self.save_checkpoints = config['save checkpoints']
        # for key, value in config.items():
        #     setattr(self, key.replace(" ", "_"), value)


class GridSearchConfig:
    # Temporary class to store configurations
    # (to be replaced with `sacred` library or similar)
    def __init__(self, config_file):
        config = json.load(open(config_file, 'r'))
        self.seed = config['seed']
        self.learning_rates = config['learning rates']
        self.batch_sizes = config['batch sizes']
        self.max_length = config['max length']
        self.n_epochs = config['n epochs']
        self.data_file = config['data file']
        self.split_file = config['split file']
        self.output_dir = config['output dir']
        self.freeze = config['freeze']
        self.save_checkpoints = config['save checkpoints']


def train_test_dev_split(data_path, split_path, cols=None):
    if cols is None:
        cols = ['idiom', 'sentence', 'label', 'transparency', 'head pos', 'corpus', 'split']
    data = pd.read_csv(data_path, sep='\t')
    split = pd.read_csv(split_path, sep='\t')
    data['split'] = split['split']
    train_data = data[data['split'] == 'train']
    test_data = data[data['split'] == 'test']
    dev_data = data[data['split'] == 'dev']
    return train_data[cols], dev_data[cols], test_data[cols]

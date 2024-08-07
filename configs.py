import json


class CrossLingualConfig:
    def __init__(self, config_file):
        config = json.load(open(config_file, 'r'))
        self.seed = config['seed']
        self.batch_size = config['batch size']
        self.max_length = config['max length']
        self.data_file = config['data file']
        self.checkpoints_dir = config['checkpoints dir']
        self.output_dir = config['output dir']


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


class FineTuneMultipleConfig:
    # Temporary class to store configurations
    # (to be replaced with `sacred` library or similar)
    def __init__(self, config_file):
        config = json.load(open(config_file, 'r'))
        self.seed = config['seed']
        self.n_models = config['n models']
        self.learning_rate = config['learning rate']
        self.batch_size = config['batch size']
        self.max_length = config['max length']
        self.n_epochs = config['n epochs']
        self.data_file = config['data file']
        self.split_dir = config['split dir']
        self.start_split = config['start split']
        self.output_dir = config['output dir']
        self.freeze = config['freeze']
        self.setting = config['setting']
        self.setting = config['setting']


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


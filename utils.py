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
    # (to be replaced with `sacred` library)
    def __init__(self, config_file):
        config = json.load(open(config_file, 'r'))
        self.seed = config['seed']
        self.learning_rate = config['learning rate']
        self.batch_size = config['batch size']
        self.max_length = config['max length']
        self.n_epochs = config['n epochs']
        self.data_dir = config['data dir']
        self.split_dir = config['split dir']
        self.model_dir = config['model dir']
        self.model_name = config['model name']
        self.output_dir = config['output dir']
        self.freeze = config['freeze']
        self.save_checkpoints = config['save checkpoints']
        # for key, value in config.items():
        #     setattr(self, key.replace(" ", "_"), value)


class GridSearchConfig:
    # Temporary class to store configurations
    # (to be replaced with `sacred` library)
    def __init__(self, config_file, batch_size):
        config = json.load(open(config_file, 'r'))
        self.seed = config['seed']
        self.learning_rate = config['learning rate']
        self.max_length = config['max length']
        self.n_epochs = config['n epochs']
        self.data_dir = config['data dir']
        self.split_dir = config['split dir']
        self.model_dir = config['model dir']
        self.output_dir = config['output dir']
        self.freeze = config['freeze']
        self.save_checkpoints = config['save checkpoints']
        self.batch_size = None

        # for key, value in config.items():
        #     setattr(self, key.replace(" ", "_"), value)


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




def read_command_line_for_fine_tuning():
    parser = argparse.ArgumentParser(description='Fine-tuning XLM-Roberta for idiomaticity detection')

    # I/O
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--split_path", type=str, required=True)
    parser.add_argument("--save_model_to", type=str, default=None)
    parser.add_argument("--save_config_to", type=str, default='.')

    # Model, data & optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--freeze", type=bool, default=False)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--setting", type=str, default="zero-shot")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--n_epochs", type=int, default=3)


    # Miscellaneous
    parser.add_argument("--save_checkpoints", type=bool, default=False)
    parser.add_argument("--seed", type=int, default=2024)

    return parser

#
#
# num_epochs = 1  # Only a few epochs for the test
# total_steps = len(train_dataloader) * num_epochs
# scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
#
# # Training loop for the learning rate range test
# model.train()
# learning_rates = []
# losses = []
#
# for batch in train_dataloader:
#     inputs, labels = batch
#     optimizer.zero_grad()
#
#     outputs = model(**inputs)
#     loss = criterion(outputs.logits, labels)
#
#     loss.backward()
#     optimizer.step()
#     scheduler.step()
#
#     # Record the learning rate and loss
#     learning_rates.append(optimizer.param_groups[0]["lr"])
#     losses.append(loss.item())
#
#     # Update the learning rate
#     for param_group in optimizer.param_groups:
#         param_group['lr'] *= 1.1  # Increase learning rate by a factor
#
# # Plot the loss vs. learning rate
# plt.plot(learning_rates, losses)
# plt.xscale('log')
# plt.xlabel('Learning Rate')
# plt.ylabel('Loss')
# plt.title('Learning Rate Range Test')
# plt.show()


import os
import random
import json
import argparse
from datetime import datetime
import torch
from torch.optim import AdamW
from transformers import XLMRobertaTokenizer
from data import IdiomDataset
from utils import make_dir
from torch.utils.data import DataLoader
from classifier import IdiomaticityClassifier
from utils import ExperimentConfig, train_test_dev_split
from trainer import IdiomaticityTrainer

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Fine-tuning XLM-Roberta for idiomaticity detection')
    parser.add_argument("--config_file", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = ExperimentConfig(args.config_file)

    # Prepare output directory
    make_dir(config.model_dir)

    # Setting up seed value
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(config.seed)

    # Setting up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setting up model type and creating model instance
    MODEL_TYPE = 'xlm-roberta-base'
    model = IdiomaticityClassifier(config)

    # Setting up tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)

    # Setting up optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)

    # Loading data
    train, dev, test = train_test_dev_split(config.data_dir, config.split_dir)

    train_set = IdiomDataset(train, tokenizer=tokenizer, max_length=config.max_length)
    test_set = IdiomDataset(test, tokenizer=tokenizer, max_length=config.max_length)
    dev_set = IdiomDataset(dev, tokenizer=tokenizer, max_length=config.max_length)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=config.batch_size, shuffle=True)

    trainer = IdiomaticityTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        test_loader=test_loader,
        val_loader=dev_loader,
        args=config
    )

    trainer.fine_tune()
    trainer.save_config()
    raise ValueError



# args = read_command_line_for_fine_tuning()
# args = args.parse_args()
#
# # Manage directory for checkpoints
# if args.save_model_to is None:
#     args.save_model_to = 'checkpoints'
# print(f"The model and/or checkpoints will be saved in the directory `{args.save_model_to}`.")
# make_dir(args.save_model_to)
#
# if args.model_name is None:
#     now = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
#     args.model_name = f"{args.setting} {now}"
# print(f"The model and/or checkpoints will be saved as `{args.model_name}`")
# print(args.__dict__)
#
# # Manage directory for configs
# if args.save_config_to is None:
#     args.save_config_to = '.'
#
# # Setting up seed value
# random.seed(args.seed)
# torch.manual_seed(args.seed)
# torch.cuda.manual_seed(args.seed)
# torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# os.environ['PYTHONHASHSEED'] = str(args.seed)
#
# # Setting up the device
# device = "cuda" if torch.cuda.is_available() else "cpu"
#
# # Setting up model type and creating model instance
# MODEL_TYPE = 'xlm-roberta-base'
# model = IdiomaticityClassifier(args)
#
# # Setting up tokenizer
# tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
#
# # Setting up optimizer
# optimizer = AdamW(model.parameters(), lr=args.learning_rate)
#
# # Loading data
# train, dev, test = train_test_dev_split(args.data_path, args.split_path)
#
# train_set = IdiomDataset(train, tokenizer=tokenizer, max_length=args.max_length)
# test_set = IdiomDataset(test, tokenizer=tokenizer, max_length=args.max_length)
# dev_set = IdiomDataset(dev, tokenizer=tokenizer, max_length=args.max_length)
#
# train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
# test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
# dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)
#
# trainer = IdiomaticityTrainer(
#     model=model,
#     optimizer=optimizer,
#     device=device,
#     train_loader=train_loader,
#     test_loader=test_loader,
#     val_loader=dev_loader,
#     args=args
# )
#
# trainer.fine_tune()
# # outputs = trainer.test_model()
# predictions, prediction_probs, true_labels = trainer.get_predictions()
# print(predictions)
# print(prediction_probs)
# print(true_labels)
# trainer.save_config('results.json')
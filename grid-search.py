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

    # add combinations of parameters
    # add fine-tuning for each combination
    EXP_DIR = "grid search results"
    make_dir(EXP_DIR)
    print("GRID SEARCH")
    batch_sizes = [4, 8, 16, 32, 64]
    for i, batch_size in enumerate(batch_sizes):
        config = ExperimentConfig('grid_search_config.json')
        config.batch_size = batch_size

        # Create and set model name
        model_name = f"lr=5e-5 batch={batch_size} setting=zero-shot"
        config.model_name = model_name
        print(f"Training model `{model_name}` for batch size of `{batch_size}`")

        # Setting up output directory
        model_dir = os.path.join(EXP_DIR, model_name)
        make_dir(model_dir)
        config.model_dir = model_dir
        config.output_dir = model_dir
        print(f"The model, the checkpoints and the config will be saved in the directory `{model_dir}`")

        # Setting up split directory
        split_name = f'split_{i}.tsv'
        split_dir = config.split_dir + split_name
        config.split_dir = split_dir
        print(f"Loading split from {config.split_dir}")

        # Setting up seed value
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ['PYTHONHASHSEED'] = str(config.seed)
        print("Random seed set.")

        # Setting up the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")

        # Setting up model type and creating model instance
        MODEL_TYPE = 'xlm-roberta-base'
        model = IdiomaticityClassifier(config)
        model.to(device)
        print("Initialized the model.")

        # Setting up tokenizer
        tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
        print("Initialized the tokenizer.")

        # Setting up optimizer
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        print("Initialized the optimizer.")

        # Loading data
        train, dev, test = train_test_dev_split(config.data_dir, config.split_dir)

        train_set = IdiomDataset(train, tokenizer=tokenizer, max_length=config.max_length)
        test_set = IdiomDataset(test, tokenizer=tokenizer, max_length=config.max_length)
        dev_set = IdiomDataset(dev, tokenizer=tokenizer, max_length=config.max_length)

        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_set, batch_size=config.batch_size, shuffle=True)
        print("Loaded data.")

        trainer = IdiomaticityTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            val_loader=dev_loader,
            args=config
        )
        print(f"Initialized training loop.")
        print()
        trainer.fine_tune()
        trainer.save_config()
        print()
        print("Finished fine-tuning.")
        print("Saved configs.")



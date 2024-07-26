import os
import random
import argparse
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
    make_dir(config.output_dir)

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
    model = IdiomaticityClassifier(model_type=MODEL_TYPE, freeze=config.freeze)
    model.to(device)
    print("Initialized the model.")

    # Setting up tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
    print("Initialized the tokenizer.")

    # Setting up optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    print("Initialized the optimizer.")

    # Loading data
    train, dev, test = train_test_dev_split(config.data_dir, config.split_file)

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
        n_epochs=config.n_epochs,
        model_name=config.model_name,
        save_checkpoints=config.save_checkpoints,
        output_dir=config.output_dir
    )

    print(f"Initialized training loop.")
    print()
    trainer.fine_tune()
    trainer.save_config()
    print()
    print("Finished fine-tuning.")
    print("Saved configs.")

import os
import json
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
from tester import IdiomaticityTester


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Fine-tuning XLM-Roberta for idiomaticity detection')
    parser.add_argument("-c", "--config_file", type=str, required=True)
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
    train, dev, test = train_test_dev_split(config.data_file, config.split_file)

    train_set = IdiomDataset(train, tokenizer=tokenizer, max_length=config.max_length)
    dev_set = IdiomDataset(dev, tokenizer=tokenizer, max_length=config.max_length)
    test_set = IdiomDataset(test, tokenizer=tokenizer, max_length=config.max_length)

    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    print("Loaded data.")

    trainer = IdiomaticityTrainer(
        model=model,
        optimizer=optimizer,
        device=device,
        train_loader=train_loader,
        val_loader=dev_loader,
        n_epochs=config.n_epochs,
        model_name=config.model_name,
        output_dir=config.output_dir
    )

    print(f"Initialized training loop.")
    print()
    trainer.fine_tune()
    trainer.save_config()
    print("Finished fine-tuning.")
    print("Saved configs.")

    if trainer.best_model is not None:
        print(f"Testing the model from checkpoint: `{trainer.best_model}`")
        checkpoint = os.path.join(config.output_dir, trainer.best_model)

        # Setting up model type and creating model instance
        model = IdiomaticityClassifier(model_type=MODEL_TYPE, freeze=config.freeze)
        model.load_state_dict(torch.load(checkpoint))
        model.to(device)
        print("Initialized the model.")

        # Setting up tester
        tester = IdiomaticityTester(
            model=model,
            device=device,
            test_loader=test_loader
        )

        # Testing and saving results
        print("Testing...")
        test_results = tester.test()
        results_path = os.path.join(config.output_dir, 'test results.json')
        json.dump(test_results, open(results_path, 'w'), indent=True)
        print(f"Test results saved to {results_path}")

    else:
        print("The model was not trained due to validation loss going up at epoch 0.")
    print("Done.")
    print("Exit.")
    print()
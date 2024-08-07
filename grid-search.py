import os
import random
import json
import argparse
from itertools import product
import torch
from torch.optim import AdamW
from transformers import XLMRobertaTokenizer
from data import IdiomDataset
from utils import make_dir
from torch.utils.data import DataLoader
from classifier import IdiomaticityClassifier
from utils import train_test_dev_split
from configs import GridSearchConfig
from trainer import IdiomaticityTrainer

if __name__ == "__main__":

    print("=============")
    print("    START")
    print('=============\n')

    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument("-c", "--config_file", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = GridSearchConfig(args.config_file)

    # Prepare output directory
    # to save experiment results
    make_dir(config.output_dir)

    # save experiments
    experiment_data = []

    # Enumerate combinations of hyperparameters
    hyperparameters = list(product(config.learning_rates, config.batch_sizes))

    print(f"The experiment includes fine-tuning {len(hyperparameters)} models with the following hyperparameters:")
    for lr, bs in hyperparameters:
        print(f"\t- learning rate: {lr}, batch size: {bs}")
    exp_time = round((len(hyperparameters) * 25)/60, 1)
    print()
    print(f"Estimated time: {exp_time} hours.")

    # Set up model type
    MODEL_TYPE = 'xlm-roberta-base'

    # Setting up tokenizer
    print("Initializing the tokenizer... ", end="")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
    print("Done.")

    # Loading data
    print(f"Loading data from {config.data_file}.")
    print(f"Loading train-dev-test split from {config.split_file}.")
    train, dev, test = train_test_dev_split(config.data_file, config.split_file)
    train_set = IdiomDataset(train, tokenizer=tokenizer, max_length=config.max_length)
    test_set = IdiomDataset(test, tokenizer=tokenizer, max_length=config.max_length)
    dev_set = IdiomDataset(dev, tokenizer=tokenizer, max_length=config.max_length)
    print("Data loaded.")

    for learning_rate in config.learning_rates:
        for batch_size in config.batch_sizes:
            print("-----------------------")
            print(f"Fine-tuning XLMRoberta")
            print("Hyperparameters:")
            print(f"\tlearning rate: {learning_rate}")
            print(f"\tbatch size: {batch_size}")
            print()

            current_experiment_data = {
                'learning rate': learning_rate,
                'batch size': batch_size,
                'split': os.path.basename(config.split_file)
            }

            # Create and set model name based on the parameters
            model_name = f"zero-shot lr={learning_rate} b={batch_size}"
            current_experiment_data['model name'] = model_name
            print(f"Model name: \"{model_name}\"")

            # Create directory for the current model
            # inside experiment directory (`config.output_dir`)
            model_path = os.path.join(config.output_dir, model_name)
            make_dir(model_path)
            print(f"The model, the checkpoints and the results will be saved \n    in `{model_path}`")
            print("-----------------------\n")

            # Setting up seed value
            print("Setting up seed value... ", end="")
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(config.seed)
            print("Done")

            # Setting up the device
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Fine-tuning on device: {device}")

            # Setting up model type and creating model instance
            print("Initializing the model...", end="")
            model = IdiomaticityClassifier(model_type=MODEL_TYPE, freeze=config.freeze)
            model.to(device)
            current_experiment_data['pretrained model'] = MODEL_TYPE
            print("Done.")

            # Setting up optimizer
            print("Initializing the optimizer... ", end="")
            optimizer = AdamW(model.parameters(), lr=learning_rate)
            print("Done.")

            #
            train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
            dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True)
            print("Done.")

            current_experiment_data['# train batches'] = len(train_loader)
            current_experiment_data['# test batches'] = len(test_loader)
            # current_experiment_data['# dev batches'] = len(dev_loader)

            trainer = IdiomaticityTrainer(
                model=model,
                optimizer=optimizer,
                device=device,
                train_loader=train_loader,
                val_loader=dev_loader,
                n_epochs=config.n_epochs,
                model_name=model_name,
                output_dir=model_path
            )

            print(f"\nInitializing training loop.\n")
            trainer.fine_tune()
            print("Done.")
            trainer.save_config()
            print("Config file saved.")

            current_experiment_data['validation loss'] = trainer.validation_loss
            current_experiment_data['validation accuracy'] = trainer.validation_accuracy

            experiment_data.append(current_experiment_data)
            print()

    # Save experiment data to `output_dir`
    out_data_path = os.path.join(config.output_dir, 'experiment_data.json')
    if os.path.exists(out_data_path):
        print("Previous experiment detected.")
        print("Added new data to the previous experiment.")
        previous_experiment = json.load(open(out_data_path, 'r'))
        for data in experiment_data:
            previous_experiment.append(data)
        json.dump(previous_experiment, open(out_data_path, 'w'), indent=True)
    else:
        json.dump(experiment_data, open(out_data_path, 'w'), indent=True)
    print(f"Experiment data saved to `{out_data_path}`")
    print("Done.")
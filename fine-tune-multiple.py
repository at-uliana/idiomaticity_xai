import os
import random
import json
import argparse
import torch
from torch.optim import AdamW
from transformers import XLMRobertaTokenizer
from data import IdiomDataset
from utils import make_dir
from torch.utils.data import DataLoader
from utils import train_test_dev_split, save_experiment_data
from configs import FineTuneMultipleConfig
from trainer import IdiomaticityTrainer
from tester import IdiomaticityTester
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig


if __name__ == "__main__":

    print("=============")
    print("    START")
    print('=============\n')

    parser = argparse.ArgumentParser(description='Fine-tune and test multiple models')
    parser.add_argument("-c", "--config_file", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = FineTuneMultipleConfig(args.config_file)

    # Prepare output directory to save checkpoints and results
    make_dir(config.output_dir)

    print(f"The experiment includes fine-tuning {config.n_models} models with the following hyperparameters:")
    print(f" - learning rate: {config.learning_rate}")
    print(f" - batch size: {config.batch_size}")

    # Setting up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fine-tuning on device: {device}")

    # Setting up model type
    MODEL_TYPE = 'xlm-roberta-base'

    # Setting up tokenizer
    print("Initializing the tokenizer... ", end="")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
    print("Done.")

    split_n = config.start_split
    experiment_data = []
    experiment_predictions = []

    for i in range(config.n_models):
        current_experiment_data = {
            "learning rate": config.learning_rate,
            "batch size": config.batch_size,
            "split": f"split_{split_n}",
        }

        model_name = f"{config.setting} lr={config.learning_rate} b={config.batch_size} split={split_n}"
        model_dir = os.path.join(config.output_dir, model_name)
        make_dir(model_dir)
        print(f"Model name: \"{model_name}\"")
        current_experiment_data["model name"] = model_name

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

        # Setting up model type and creating model instance
        print("Initializing the model...", end="")

        model = XLMRobertaForSequenceClassification.from_pretrained(MODEL_TYPE,
                                                                    num_labels=2,
                                                                    output_hidden_states=False,
                                                                    output_attentions=False
                                                                    )

        model.to(device)
        print("Done.")

        # Setting up optimizer
        print("Initializing the optimizer... ", end="")
        optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        print("Done.")

        # Loading test-dev-test split file
        split_file = os.path.join(config.split_dir, f"split_{split_n}.tsv")
        print(f"Loading data from {config.data_file}.")
        print(f"Loading train-dev-test split from {split_file}.")

        train, dev, test = train_test_dev_split(config.data_file, split_file)
        train_set = IdiomDataset(train, tokenizer=tokenizer, max_length=config.max_length)
        dev_set = IdiomDataset(dev, tokenizer=tokenizer, max_length=config.max_length)
        test_set = IdiomDataset(test, tokenizer=tokenizer, max_length=config.max_length)

        train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
        dev_loader = DataLoader(dev_set, batch_size=config.batch_size, shuffle=True)
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

        current_experiment_data['# train batches'] = len(train_loader)
        current_experiment_data['# test batches'] = len(test_loader)
        current_experiment_data['# dev batches'] = len(dev_loader)
        print("Done.")

        trainer = IdiomaticityTrainer(
            model=model,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            val_loader=dev_loader,
            n_epochs=config.n_epochs,
            model_name=model_name,
            output_dir=model_dir
        )

        print(f"\nInitializing training loop.\n")
        trainer.fine_tune()
        print("Fine-tuning completed.")
        trainer.save_config()
        print("Config file saved.")
        print('----------------------\n')

        if trainer.best_model is not None:
            print(f"Testing the model from checkpoint: `{trainer.best_model}`")

            # Add validation loss from the best epoch:
            vals = trainer.validation_loss[1:]  # don't consider validation before fine-tuning
            epoch_loss = vals[trainer.best_epoch]
            current_experiment_data['validation loss'] = epoch_loss
            current_experiment_data['best epoch'] = trainer.best_epoch

            # Initialize model from config
            model_config = XLMRobertaConfig.from_pretrained(MODEL_TYPE)
            config.num_labels = 2
            model = XLMRobertaForSequenceClassification(model_config)

            # Load state_dict
            checkpoint_path = os.path.join(model_dir, trainer.best_model)
            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint)

            model.to(device)

            # Setting up tester
            tester = IdiomaticityTester(
                model=model,
                device=device,
                test_loader=test_loader
            )

            # Testing
            print("Testing...")
            test_results = tester.test()
            test_results['model checkpoint'] = trainer.best_model

            # Save intermediate results
            results_path = os.path.join(model_dir, f'test_results.json')
            json.dump(test_results, open(results_path, 'w'), indent=True)
            current_experiment_data['test results'] = results_path
            print(f"Test results saved to {results_path}")

            # Copy test results also to the experiment data
            for key in test_results:
                if key not in ('predictions', 'true labels', 'probabilities'):
                    current_experiment_data[key] = test_results[key]

            experiment_predictions.append(
                {
                    'split': split_n,
                    'predictions': test_results['predictions'],
                    'true labels': test_results['true labels'],
                    'probabilities': test_results['probabilities']
                }
            )

        else:
            print("The model was not trained due to validation loss going up at after the first epoch.")

        split_n += 1
        experiment_data.append(current_experiment_data)

    # Save all the data
    out_data_path = os.path.join(config.output_dir, 'experiment_data.json')
    save_experiment_data(experiment_data, out_data_path)
    print(f"Experiment data saved to `{out_data_path}`\n")

    # Save predictions
    out_data_path = os.path.join(config.output_dir, 'predictions.json')
    save_experiment_data(experiment_predictions, out_data_path)
    print(f"Test results (predictions and probabilities) saved to `{out_data_path}`\n")
    print("Done.")

    tokenizer.save_pretrained(config.output_dir)

    print("Exit.")

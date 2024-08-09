import os
import json
import random
import argparse
import torch
import pandas as pd
from transformers import XLMRobertaTokenizer
from data import IdiomDataset
from utils import make_dir, save_experiment_data
from torch.utils.data import DataLoader
from configs import CrossLingualConfig
from tester import IdiomaticityTester
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig


if __name__ == "__main__":

    print("=============")
    print("    START")
    print('=============\n')

    parser = argparse.ArgumentParser(description='Hyperparameter optimization')
    parser.add_argument("-c", "--config_file", type=str, required=True)
    args = parser.parse_args()

    # Load configuration file
    config = CrossLingualConfig(args.config_file)

    # Prepare output directory to save results
    make_dir(config.output_dir)

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

    # Setting up model type
    MODEL_TYPE = 'xlm-roberta-base'

    # Setting up tokenizer
    print("Initializing the tokenizer... ", end="")
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
    print("Done.")

    # Load data
    data = pd.read_csv(config.data_file, sep='\t')
    test_set = IdiomDataset(data, tokenizer=tokenizer, max_length=config.max_length)
    results = []

    for model_name in os.listdir(config.checkpoints_dir):
        print(f"Testing checkpoint {model_name}")
        print("Loading data...")
        test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False)

        # Initialize model from config
        model_config = XLMRobertaConfig.from_pretrained(MODEL_TYPE)
        config.num_labels = 2
        model = XLMRobertaForSequenceClassification(model_config)

        # Load state_dict
        checkpoint_path = os.path.join(config.checkpoints_dir, model_name)
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
        print(model_name)
        test_results = tester.test()

        # Save results
        test_results['model checkpoint'] = model_name
        results.append(test_results)
        model_name = model_name.split(".")[0]
        results_path = os.path.join(config.output_dir, f'test_results_{model_name}.json')
        json.dump(test_results, open(results_path, 'w'), indent=True)

    # Save all data
    print("Save experiment data...", end=' ')
    path = os.path.join(config.output_dir, 'test_results.json')
    save_experiment_data(results, path)
    print("done.")
    print("Exit.")


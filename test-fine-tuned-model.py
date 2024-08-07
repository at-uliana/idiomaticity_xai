import json
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
from utils import train_test_dev_split
from configs import ExperimentConfig
from tester import IdiomaticityTester

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Testing fine-tuned models')
    parser.add_argument("-d", "--data_file", type=str, required=True)
    parser.add_argument("-s", "--split_file", type=str, required=True)

    args = parser.parse_args()

    # Setting up the device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Setting up model type and creating model instance
    MODEL_TYPE = 'xlm-roberta-base'
    model = IdiomaticityClassifier(model_type=MODEL_TYPE)

    # Load the state dictionary into the model
    model.load_state_dict(torch.load(args.model_file))

    model.to(device)
    print("Initialized the model.")

    # Setting up tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)
    print("Initialized the tokenizer.")

    # Loading data
    _, _, test = train_test_dev_split(args.data_file, args.split_file)

    test_set = IdiomDataset(test, tokenizer=tokenizer, max_length=args.max_length)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    # Initalize testing
    tester = IdiomaticityTester(model=model, device=device, test_loader=test_loader)
    results = tester.test()
    path = os.path.join(args.output_file, 'test.json')
    json.dump(results, open(path, 'w'), indent=True)


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

    config = ExperimentConfig('grid search config.json')

    # Prepare output directory
    make_dir(config.model_dir)

    # add combinations of parameters
    # add fine-tuning for each combination
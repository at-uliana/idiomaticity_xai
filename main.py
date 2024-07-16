from transformers import XLMRobertaTokenizer
from data import IdiomDataset
import torch
import os, random
from pprint import pprint
import numpy as np
from torch.utils.data import DataLoader
from classifier import IdiomaticityClassifier
from utils import Args, train_test_dev_split
from trainer import IdiomaticityTrainer
from torch.optim import AdamW


# Reproducibility
seed_value = 42
random.seed(seed_value)
np.random.seed(seed_value)
torch.manual_seed(seed_value)
torch.cuda.manual_seed(seed_value)
torch.cuda.manual_seed_all(seed_value)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed_value)

# Setting up the device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

# Setting up arguments (TODO: replace with ArgParser)
args = Args(data_path='../../data/magpie+semeval+epie.tsv',
            split_path='../../data/split/zero-shot/split_0.tsv',
            batch_size=8, learning_rate=5e-5, n_epochs=2)

# Setting up model type and creating model instance
MODEL_TYPE = 'xlm-roberta-base'
model = IdiomaticityClassifier(args)

# Setting up tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)

# Setting up optimizer
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

# Loading data
train, dev, test = train_test_dev_split(args.data_path, args.split_path)

train_set = IdiomDataset(train, tokenizer=tokenizer)
test_set = IdiomDataset(test, tokenizer=tokenizer)
dev_set = IdiomDataset(dev, tokenizer=tokenizer)

train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=True)

print(f"Number of batches in `train`: {len(train_loader)}.")
print(f"Number of batches in `test`: {len(test_loader)}.")
print(f"Number of batches in `dev`: {len(dev_loader)}.")
print()

trainer = IdiomaticityTrainer(
    model=model,
    optimizer=optimizer,
    device=device,
    train_loader=train_loader,
    test_loader=test_loader,
    val_loader=dev_loader,
    args=args
)

trainer.fine_tune()
outputs = trainer.test_model()
trainer.save_config('results.json')
pprint(trainer.results)

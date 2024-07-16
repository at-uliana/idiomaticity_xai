from transformers import XLMRobertaTokenizer
from data import IdiomDataset
import torch
import json
import random
import numpy as np
import os
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
args = Args(batch_size=8, learning_rate=5e-5)

# Setting up model type and creating model instance
MODEL_TYPE = 'xlm-roberta-base'
model = IdiomaticityClassifier(args)

# Setting up tokenizer
tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)

# Setting up optimizer
optimizer = AdamW(model.parameters(), lr=args.learning_rate)

# Loading data
data_path = '../../data/magpie+semeval+epie.tsv'
split_path = '../../data/split/zero-shot/split_0.tsv'
train, dev, test = train_test_dev_split(data_path, split_path)

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
json.dump(trainer.results, open('results.json', 'w'), indent=True)


# results = train_loop(model, train_loader, dev_loader, epochs=2, optimizer=optimizer, device=device, args=args)

# print("Batch example:")
# input_ids, attn_mask, labels = next(iter(test_loader))
# batch = next(iter(test_loader))

# print(input_ids, att_mask, label)


# initial_loss = (dev_loader, model)
# print(f"Loss before: {loss}")
# i = 0
# for batch in iter(test_loader):
#     loss = train_batch(model, batch, optimizer, args)
#     # print(output)
#     print(f"Batch {i}: {loss}")
#     i += 1
#     if i == 1:
#         break

# loss = evaluate(model, validation_loader=dev_loader)
# print(f"Loss after: {loss}")

# epoch_model_path = f'models/model_10_batches.bin'
# torch.save(model.state_dict(), epoch_model_path)

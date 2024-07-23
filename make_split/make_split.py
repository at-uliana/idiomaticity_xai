import os
import logging
import argparse
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(format="%(levelname)s\t%(name)s\t%(message)s")


def make_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


parser = argparse.ArgumentParser(description='Make data split for zero-shot, one-shot and random setting')

parser.add_argument("--datadir", type=str, required=True)
parser.add_argument("--setting", type=str, required=True)
parser.add_argument("--outdir", type=str, required=True)
parser.add_argument("--n_splits", type=int, required=True)
parser.add_argument("--seed", type=int, required=False, default=2025)

args = parser.parse_args()

# Set random seed
np.random.seed(args.seed)

# Create directory to save splits
make_dir(folder=args.outdir)

# Import data
data = pd.read_csv(args.datadir, sep='\t')

# Print data stats
idioms = data['idiom'].unique()
print(f"Number of idioms: {len(idioms)}")
print(f"Data size: {len(data)}")

# Perform splitting for zero-shot setting
if args.setting == 'zero-shot':
    train_size_total = 0
    test_size_total = 0
    dev_size_total = 0

    for i in range(args.n_splits):
        data['split'] = 'train'

        # Select 200 idioms for testing
        idioms_test = np.random.choice(idioms, size=250, replace=False)

        # Move all sentences with the selected idioms to the test set
        data.loc[data['idiom'].isin(idioms_test), 'split'] = 'test'

        # Select 100 idioms for validation (disjoint from the test set)
        idioms_train = data.loc[data['split'] == 'train', 'idiom'].unique()
        idioms_dev = np.random.choice(idioms_train, size=250, replace=False)

        # Move all sentences with the selected idioms to the dev set
        data.loc[data['idiom'].isin(idioms_dev), 'split'] = 'dev'

        name = f"split_{i}.tsv"
        path = os.path.join(args.outdir, name)
        data['split'].to_csv(path, index=False)

        train_size = (data['split'] == 'train').sum()
        test_size = (data['split'] == 'test').sum()
        dev_size = (data['split'] == 'dev').sum()

        print(f"Split# {i}")
        print(f"Train size: {train_size}")
        print(f"Test size: {test_size}")
        print(f"Dev size: {dev_size}")
        print()

        train_size_total += train_size
        test_size_total += test_size
        dev_size_total += dev_size
    print("-------------------------")
    print(f"Average train size: {round(train_size_total / 10)}")
    print(f"Average test size: {round(test_size_total / 10)}")
    print(f"Average dev size: {round(dev_size_total / 10)}")

elif args.setting == 'one-shot':
    train_size_total = 0
    test_size_total = 0
    dev_size_total = 0

    for i in range(args.n_splits):
        data['split'] = 'train'

        # Select 200 idioms for testing
        idioms_test = np.random.choice(idioms, size=200, replace=False)

        # Move all sentences with the selected idioms to the dev set
        data.loc[data['idiom'].isin(idioms_test), 'split'] = 'dev'

        name = f"split_{i}.tsv"
        path = os.path.join(args.outdir, name)
        data['split'].to_csv(path, index=False)

        train_size = (data['split'] == 'train').sum()
        test_size = (data['split'] == 'test').sum()
        dev_size = (data['split'] == 'dev').sum()

        print(f"Split# {i}")
        print(f"Train size: {train_size}")
        print(f"Dev size: {dev_size}")
        print()
        train_size_total += train_size
        dev_size_total += dev_size

    print("-------------------------")
    print(f"Average train size: {round(train_size_total / 10)}")
    print(f"Average test size: {round(test_size_total / 10)}")
    print(f"Average dev size: {round(dev_size_total / 10)}")
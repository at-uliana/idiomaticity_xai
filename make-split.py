import os
import logging
import argparse
import pandas as pd
import numpy as np
from utils import make_dir


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Make data split for zero-shot, one-shot and random setting')

    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--setting", type=str, required=True)
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--n_splits", type=int, required=True)
    parser.add_argument("--seed", type=int, required=False, default=2025)
    parser.add_argument("--generate_report", type=bool, required=False, default=False)

    # arguments valid only for `random` setting
    parser.add_argument("--test_size", type=int, required=False, default=5000)
    parser.add_argument("--dev_size", type=int, required=False, default=5000)

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create directory to save splits
    make_dir(dir=args.outdir)

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

    elif args.setting == 'random':
        print("Split size for all splits:")
        print(f"Train size: {len(data) - args.test_size - args.dev_size}")
        print(f"Test size: {args.test_size}")
        print(f"Dev size: {args.ev_size}")
        print("--------------------------")

        for i in range(args.n_splits):
            avg_n_idioms_train = 0
            avg_n_idioms_test = 0
            avg_n_idioms_dev = 0

            data['split'] = 'train'

            # Choose test set
            test = np.random.choice(data.index, size=args.test_size, replace=False)
            data.loc[test, 'split'] = 'test'

            # Choose dev set
            train_data = data[data['split'] == 'train']
            dev = np.random.choice(train_data.index, size=args.dev_size, replace=False)
            data.loc[dev, 'split'] = 'dev'

            # Save split
            name = f"split_{i}.tsv"
            path = os.path.join(args.outdir, name)
            data['split'].to_csv(path, index=False)

            train_size = data[data['split'] == 'train']['idiom'].nunique()
            test_size = data[data['split'] == 'test']['idiom'].nunique()
            dev_size = data[data['split'] == 'dev']['idiom'].nunique()

            avg_n_idioms_train += train_size
            avg_n_idioms_test += test_size
            avg_n_idioms_dev += dev_size

            print(f"Split #{i}")
            print(f"Number of idioms in train: {train_size}")
            print(f"Number of idioms in test: {test_size}")
            print(f"Number of idioms in dev: {dev_size}")
            print()
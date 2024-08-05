import os
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

    # arguments valid only for `random` setting
    parser.add_argument("--test_size", type=int, required=False, default=6500)
    parser.add_argument("--dev_size", type=int, required=False, default=6500)

    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)

    # Create directory to save splits
    make_dir(dir=args.outdir)

    # Perform splitting for zero-shot setting
    if args.setting == 'zero-shot' or args.setting == 'zeroshot':

        print("---------------------------------------")
        print("Creating zero-shot train-test-dev splits")
        print("---------------------------------------")

        train_size_total = 0
        test_size_total = 0
        dev_size_total = 0

        # Import data
        data = pd.read_csv(args.datadir, sep='\t')

        # Print data stats
        idioms = data['idiom'].unique()
        print(f"Number of idioms: {len(idioms)}")
        print(f"Data size: {len(data)}")

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

            train_data = data[data['split'] == 'train']
            test_data = data[data['split'] == 'test']
            dev_data = data[data['split'] == 'dev']

            if args.verbose:
                print(f"Split# {i}")
                print(f"Train size: {len(train_data)}")
                print(f"Test size: {len(test_data)}")
                print(f"Dev size: {len(dev_data)}")
                print()

                print(f"N idioms in train: {train_data['idiom'].nunique()}")
                print(f"N idioms in test: {test_data['idiom'].nunique()}")
                print(f"N idioms in dev: {dev_data['idiom'].nunique()}")

            train_size_total += len(train_data)
            test_size_total += len(test_data)
            dev_size_total += len(dev_data)

        print("-------------------------")
        print(f"Average train size: {round(train_size_total / args.n_splits)}")
        print(f"Average test size: {round(test_size_total / args.n_splits)}")
        print(f"Average dev size: {round(dev_size_total / args.n_splits)}")

    elif args.setting == 'one-shot' or args.setting == "oneshot":
        print(args)
        print("---------------------------------------")
        print("Creating one-shot train-test-dev splits")
        print("---------------------------------------")
        train_size_total = 0
        test_size_total = 0
        dev_size_total = 0

        # Import data
        data = pd.read_csv(args.datadir, sep='\t')
        print(f"Total idioms: {data['idiom'].nunique()}")

        def len_set(value):
            return len(set(value))

        table = data.groupby("idiom").agg({"sentence": len_set}).reset_index()
        idioms = table[table['sentence'] > 1]['idiom'].tolist()
        print(f"Number of idioms: {len(idioms)}")
        print(f"Data size: {len(data)}")

        for i in range(args.n_splits):
            idioms_test = np.random.choice(idioms, 250, replace=False)
            idioms_train = [idiom for idiom in idioms if idiom not in idioms_test]
            data['split'] = 'train'
            data.loc[data['idiom'].isin(idioms_test), 'split'] = 'test'

            idioms_dev = np.random.choice(idioms_train, 250, replace=False)
            data.loc[data['idiom'].isin(idioms_dev), 'split'] = 'dev'

            test_data = data[data['split'] == 'test']
            dev_data = data[data['split'] == 'dev']

            back_to_training = []
            for name, group in test_data.groupby('idiom'):
                index = group.sample(1).index.values[0]
                back_to_training.append(index)

            for name, group in dev_data.groupby('idiom'):
                index = group.sample(1).index.values[0]
                back_to_training.append(index)

            data.loc[back_to_training, 'split'] = 'train'

            # Write into the output directory
            name = f"split_{i}.tsv"
            path = os.path.join(args.outdir, name)
            data['split'].to_csv(path, index=False)

            # Print stats
            train_data = data[data['split'] == 'train']
            test_data = data[data['split'] == 'test']
            dev_data = data[data['split'] == 'dev']

            if args.verbose:
                print(f"Train size: {len(train_data)}")
                print(f"Test size: {len(test_data)}")
                print(f"Dev size: {len(dev_data)}")

                print(f"N idioms in train: {train_data['idiom'].nunique()}")
                print(f"N idioms in test: {test_data['idiom'].nunique()}")
                print(f"N idioms in dev: {dev_data['idiom'].nunique()}")
                print()

            train_size_total += len(train_data)
            test_size_total += len(test_data)
            dev_size_total += len(dev_data)

        print("-------------------------")
        print(f"Average train size: {round(train_size_total / args.n_splits)}")
        print(f"Average test size: {round(test_size_total / args.n_splits)}")
        print(f"Average dev size: {round(dev_size_total / args.n_splits)}")

    elif args.setting == 'random':

        print("---------------------------------------")
        print("Creating random train-test-dev splits")
        print("---------------------------------------")

        data = pd.read_csv(args.datadir, sep='\t')
        idioms = data['idiom'].nunique()
        print(f"Total idioms: {idioms}")

        print("Split size for all splits:")
        print(f"Train size: {len(data) - args.test_size - args.dev_size}")
        print(f"Test size: {args.test_size}")
        print(f"Dev size: {args.dev_size}")
        print("--------------------------")

        avg_n_idioms_train = 0
        avg_n_idioms_test = 0
        avg_n_idioms_dev = 0

        for i in range(args.n_splits):

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

        print()
        print(f"Average number of idioms in train: {round(avg_n_idioms_train/idioms)}")
        print(f"Average number of idioms in test: {round(avg_n_idioms_test/idioms)}")
        print(f"Average number of idioms in dev {round(avg_n_idioms_dev/idioms)}")
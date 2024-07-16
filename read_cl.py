import argparse

parser = argparse.ArgumentParser(description='Fine-tuning XLM-Roberta for idiomaticity detection')
parser.add_argument("-d", "--data-path", help='data folder')
parser.add_argument("-s", "--split-path", help='folder with train-test-dev split')
parser.add_argument("-lr", "--learning-rate", help='learning rate', default=1e-5)
parser.add_argument("--seed", help='seed value', default=24)


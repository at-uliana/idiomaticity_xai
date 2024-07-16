from torch.utils.data import DataLoader, Dataset
from transformers import XLMRobertaTokenizer
from utils import train_test_dev_split

MODEL_TYPE = 'xlm-roberta-base'


class IdiomDataset(Dataset):

    def __init__(self, data, tokenizer, text_col='sentence', label_col='label', max_length=128):
        self.data = data
        self.max_length = max_length
        self.tokenizer = tokenizer

    def __getitem__(self, i):
        text = self.data.iloc[i]['sentence']
        label = self.data.iloc[i]['label']
        tokens = self.tokenizer(text,
                                padding='max_length',
                                truncation=True,
                                max_length=self.max_length,
                                return_tensors='pt')
        #         output = (tokens['input_ids'][0], tokens['attention_mask'][0], label)
        return tokens['input_ids'][0], tokens['attention_mask'][0], label
        # return tokens

    def __str__(self):
        return f"<IdiomDataset ({len(self.data)} sents, {self.data['idiom'].nunique()} idioms)>"

    def __repr__(self):
        return str(self)

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    data_path = '../../data/magpie+semeval+epie.tsv'
    split_path = '../../data/split/zero-shot/split_0.tsv'
    train, dev, test = train_test_dev_split(data_path, split_path)

    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_TYPE)

    train_set = IdiomDataset(train, tokenizer=tokenizer)
    test_set = IdiomDataset(test, tokenizer=tokenizer)
    dev_set = IdiomDataset(dev, tokenizer=tokenizer)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8, shuffle=True)
    dev_loader = DataLoader(dev_set, batch_size=8, shuffle=True)

    print(f"Number of batches in `train`: {len(train_loader)}.")
    print(f"Number of batches in `test`: {len(test_loader)}.")
    print(f"Number of batches in `dev`: {len(dev_loader)}.")
    print()

    print("Batch example:")
    input_ids, att_mask, label = next(iter(test_loader))
    print(input_ids, att_mask, label)
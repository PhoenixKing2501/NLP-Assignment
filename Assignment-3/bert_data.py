import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


# BERT Dataset
class BERT_Dataset(Dataset):
    def __init__(self, sentences, tokenizer, sequence_length):
        self.sentences = sentences["text"].tolist()
        self.labels = sentences["label"].tolist()
        # self.sentences = sentences["text"]
        # self.labels = sentences["label"]
        self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer)
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        # Tokenize sentence
        encoding = self.tokenizer(sentence, return_tensors="pt",
                                  padding="max_length",
                                  truncation=True,
                                  max_length=self.sequence_length)

        encoding['input_ids'] = encoding['input_ids'].squeeze(0)
        encoding['attention_mask'] = encoding['attention_mask'].squeeze(0)

        return encoding, torch.tensor(label)

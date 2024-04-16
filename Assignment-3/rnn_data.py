import numpy as np
import torch
from gensim.models import Word2Vec
from torch.utils.data import Dataset


def get_vectorOOV(w2v_model, word, vector_size):
    try:
        return np.array(w2v_model.wv[word])
    except KeyError:
        return np.zeros((vector_size,))


# Neural Network Dataset
class NN_Dataset(Dataset):
    def __init__(self, sentences, w2v_model, output_dim):
        self.sentences = sentences["text"].tolist()
        self.labels = sentences["label"].tolist()
        self.w2v_model = Word2Vec.load(w2v_model)
        self.vector_size = self.w2v_model.vector_size
        self.output_dim = output_dim

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        _label = self.labels[idx]

        # Convert sentence to average word embeddings
        embedding = np.mean([get_vectorOOV(self.w2v_model, word, self.vector_size)
                            for word in sentence], axis=0)

        # Convert label to one-hot encoding
        label = np.zeros(self.output_dim)
        label[_label] = 1

        return torch.tensor(embedding, dtype=torch.float), \
            torch.tensor(label, dtype=torch.float)
# END NNDataset


# Recurrent Neural Network Dataset (for both RNN and LSTM)
class RNN_Dataset(Dataset):
    def __init__(self, sentences, w2v_model, sequence_length, output_dim):
        self.sentences = sentences["text"].tolist()
        self.labels = sentences["label"].tolist()
        self.w2v_model = Word2Vec.load(w2v_model)
        self.vector_size = self.w2v_model.vector_size
        self.sequence_length = sequence_length
        self.output_dim = output_dim

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        _label = self.labels[idx]

        # Convert sentence to word embeddings
        embeddings = [get_vectorOOV(self.w2v_model, word, self.vector_size)
                      for word in sentence]

        # Pad or truncate the sequence to a fixed length
        if len(embeddings) < self.sequence_length:
            embeddings += [np.zeros_like(embeddings[0])
                           for _ in range(self.sequence_length - len(embeddings))]
        else:
            embeddings = embeddings[:self.sequence_length]

        embeddings = np.array(embeddings)

        label = np.zeros(self.output_dim)
        label[_label] = 1

        return torch.tensor(embeddings, dtype=torch.float), \
            torch.tensor(label, dtype=torch.float)
# END RNNDataset

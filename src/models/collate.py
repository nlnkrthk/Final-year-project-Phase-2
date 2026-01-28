import torch
from torch.nn.utils.rnn import pad_sequence

def collate_attention(batch):
    audios, texts, labels = zip(*batch)

    audios = pad_sequence(audios, batch_first=True)
    texts = pad_sequence(texts, batch_first=True)

    labels = torch.tensor(labels)

    return audios, texts, labels

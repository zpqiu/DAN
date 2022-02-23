import torch.nn as nn
import numpy as np
import torch


class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, embed_size=512):
        super().__init__(vocab_size, embed_size, padding_idx=0)


def create_embedding_layer(weight_path, vocab_size, embed_size):
    if weight_path == "":
        return TokenEmbedding(vocab_size, embed_size=embed_size)
    weights = np.load(weight_path)
    weights = torch.tensor(weights).float()
    return nn.Embedding.from_pretrained(weights)

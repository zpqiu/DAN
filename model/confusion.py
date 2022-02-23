# -*- encoding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2019-04-19

Confusion difficulty
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention.single import Attention


class ConfusionModule(nn.Module):
    def __init__(self, batch_size, docset_size, seq_length, encode_size, dropout):
        """
        initial function
        :param batch_size:
        :param docset_size:
        :param seq_length:
        :param embedding_size:
        :param dropout:
        """
        super().__init__()
        self.attention = Attention()

        self.batch_size, self.docset_size, self.seq_length, self.encode_size = \
            batch_size, docset_size, seq_length, encode_size

        self.dropout = nn.Dropout(p=dropout)

        self.fusion = nn.Sequential(
            nn.Linear(encode_size * 2, encode_size),
            nn.ReLU()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(encode_size * 8, encode_size * 2),
            nn.ReLU()
        )

    def forward(self, x1, x2, x1_len, x2_len):
        """
        :param x1: shape: [batch_size, seq_length, encode_size]
        :param x1: shape: [batch_size, 4, seq_length, encode_size]
        :param x1_len: shape: [batch_size, ]
        :param x2_len: shape: [batch_size, 4]
        :return: shape: [batch_size, encode_size*2]
        """
        # Masking..
        # shape: [batch_size * 4, docset_size]
        x1_len = x1_len.repeat(1, 4).view(-1, self.docset_size)
        x2_len = x2_len.view(-1, self.docset_size)

        # shape: [batch_size*4, docset_size*seq_length, docset_size*seq_length]
        mask = self._create_mask(x1_len, x2_len)

        # Attention.
        x1_repeat = x1.repeat(1, 4, 1, 1).view(-1, self.docset_size, self.seq_length, self.encode_size)

        # shape: [batch_size*4, docset_size*seq_length, encode_size]
        x1_repeat = x1_repeat.view(-1, self.docset_size*self.seq_length, self.encode_size)
        x2 = x2.view(-1, self.docset_size*self.seq_length, self.encode_size)

        # shape: [batch_size*4, docset_size*seq_length, encode_size]
        x1_attend, attns = self.attention(query=x1_repeat,
                                      key=x2,
                                      value=x2,
                                      mask=mask,
                                      dropout=self.dropout)

        # Fuse.
        # shape: [batch_size * 4, docset_size * seq_length, encode_size*2]
        x1_cat = torch.cat([x1_repeat-x1_attend, x1_repeat*x1_attend], dim=-1)
        # shape: [batch_size * 4, docset_size * seq_length, encode_size]
        x1_fusion = self.fusion(x1_cat)

        # shape: [batch_size * 4, encode_size]
        x1_mean = torch.mean(x1_fusion, dim=1)
        x1_max = torch.max(x1_fusion, dim=1)[0]

        output = torch.cat([x1_mean.view(self.batch_size, -1), x1_max.view(self.batch_size, -1)], dim=-1)
        output = self.output_layer(output)
        return output, attns

    def _create_mask(self, x1_len, x2_len):
        """
        :param x1_len: see forward
        :param x2_len: see forward
        :return: shape: [batch_size*4, docset_size*seq_length, docset_size*seq_length]
        """
        # shape: [batch_size * 4 * docset_size]
        x1_len = x1_len.view(-1)
        x2_len = x2_len.view(-1)

        # shape: [batch_size * 4 * docset_size, seq_length]
        x1_mask = self._length_to_mask(x1_len, self.seq_length)
        x2_mask = self._length_to_mask(x2_len, self.seq_length)

        # shape: [batch_size * 4, docset_size*seq_length, 1]
        x1_mask = x1_mask.view(self.batch_size*4, -1).view(self.batch_size*4, 1, -1).transpose(-2, -1)
        # shape: [batch_size*4, 1, docset_size*seq_length]
        x2_mask = x2_mask.view(self.batch_size*4, -1).view(self.batch_size*4, 1, -1)

        return x1_mask * x2_mask

    def _length_to_mask(self, length, max_len=None, dtype=None):
        """length: B.
        return B x max_len.
        If max_len is None, then max of length will be used.
        """
        assert len(length.shape) == 1, 'Length shape should be 1 dimensional.'
        max_len = max_len or length.max().item()
        mask = torch.arange(max_len, device=length.device,
                            dtype=length.dtype).expand(len(length), max_len) < length.unsqueeze(1)
        if dtype is not None:
            mask = torch.as_tensor(mask, dtype=dtype, device=length.device)
        return mask

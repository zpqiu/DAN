# encoding:utf-8

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.attention.single import Attention


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length,
                 batch_first=False, num_layers=1, bidirectional=False, dropout=0.2):
        """
        :param input_size: int, embedding size
        :param hidden_size:
        :param seq_length:
        :param batch_first:
        :param num_layers:
        :param bidirectional:
        :param dropout:
        """
        super(LSTM, self).__init__()

        self.rnn = nn.LSTM(input_size=input_size,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           bidirectional=bidirectional,
                           batch_first=batch_first)
        self.reset_params()
        self.dropout = nn.Dropout(p=dropout)
        self.seq_length = seq_length

    def reset_params(self):
        for i in range(self.rnn.num_layers):
            nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}'))
            nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}'))
            nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}'), val=0)
            nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}'), val=0)
            getattr(self.rnn, f'bias_hh_l{i}').chunk(4)[1].fill_(1)

            if self.rnn.bidirectional:
                nn.init.orthogonal_(getattr(self.rnn, f'weight_hh_l{i}_reverse'))
                nn.init.kaiming_normal_(getattr(self.rnn, f'weight_ih_l{i}_reverse'))
                nn.init.constant_(getattr(self.rnn, f'bias_hh_l{i}_reverse'), val=0)
                nn.init.constant_(getattr(self.rnn, f'bias_ih_l{i}_reverse'), val=0)
                getattr(self.rnn, f'bias_hh_l{i}_reverse').chunk(4)[1].fill_(1)

    def forward(self, x):
        """
        encode
        :param x: Tuple(data, data_len), data shape: [batch_size, seq_len, embedding_size],
                  data_len shape: [batch_size, ]
        :return: Tuple, first shape: [batch_size, seq_len, embedding_size*2],
                        second shape: [batch_size, embedding_size*2]
        """
        x, x_len = x
        x = self.dropout(x)

        x_len_sorted, x_idx = torch.sort(x_len, descending=True)
        x_sorted = x.index_select(dim=0, index=x_idx)
        _, x_ori_idx = torch.sort(x_idx)

        x_packed = nn.utils.rnn.pack_padded_sequence(x_sorted, x_len_sorted, batch_first=True)
        x_packed, (h, c) = self.rnn(x_packed)

        x = nn.utils.rnn.pad_packed_sequence(x_packed, total_length=self.seq_length, batch_first=True)[0]
        x = x.index_select(dim=0, index=x_ori_idx)
        h = h.permute(1, 0, 2).contiguous().view(-1, h.size(0) * h.size(2)).squeeze()
        h = h.index_select(dim=0, index=x_ori_idx)

        return x, h


class Encoder(nn.Module):
    def __init__(self, batch_size, seq_length, embedding_size, hidden_size, dropout):
        """
        :param batch_size: int
        :param seq_length: int
        :param embedding_size: int
        :param hidden_size: int
        :param dropout: float
        """
        super().__init__()
        self.lstm = LSTM(input_size=embedding_size,
                         hidden_size=hidden_size,
                         seq_length=seq_length,
                         batch_first=True,
                         bidirectional=True,
                         dropout=dropout)

        self.attention = Attention()

        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_size*8, hidden_size*2),
            nn.Tanh()
        )

        self.dropout = nn.Dropout(p=dropout)

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

    def forward(self, qa, qa_len, docs, doc_lens):
        """
        encode
        :param qa: shape: [batch_size, seq_length, embedding_size]
        :param qa_len: shape: [batch_size, ]
        :param docs: shape: [batch_size, docset_size, seq_length, embedding_size]
        :param doc_lens: shape: [batch_size, docset_size]
        :return: shape: [batch_size, docset_size, seq_length, hidden_size*2]
        """
        batch_size = qa.size(0)
        # LSTM layer
        qa = qa.view(-1, self.seq_length, self.embedding_size)
        qa_len = qa_len.view(-1)

        qa_lstm, _ = self.lstm((qa, qa_len))

        docs = docs.view(-1, self.seq_length, self.embedding_size)
        doc_lens = doc_lens.view(-1)

        # [batch_size * docset_size, seq_length, hidden_size*2]
        doc_lstm, _ = self.lstm((docs, doc_lens))

        # Cross Attention
        # shape: [batch_size, docset_size * seq_length, seq_length]
        mask = self._create_mask(doc_lens, qa_len, batch_size)

        # shape: [batch_size, docset_size * seq_length, hidden_size * 2]
        doc_lstm = doc_lstm.view(batch_size, -1, self.hidden_size*2)
        # print(doc_lstm.size(), qa_lstm.size(), mask.size())
        doc_attend, _ = self.attention(query=doc_lstm,
                                       key=qa_lstm,
                                       value=qa_lstm,
                                       mask=mask,
                                       dropout=self.dropout)

        doc_encode = self.fusion_layer(torch.cat([doc_lstm, doc_attend, doc_lstm * doc_attend, doc_lstm - doc_attend], dim=-1))

        mask = mask.transpose(-1, -2)
        qa_attend, _ = self.attention(query=qa_lstm,
                                      key=doc_lstm,
                                      value=doc_lstm,
                                      mask=mask,
                                      dropout=self.dropout)

        qa_encode = self.fusion_layer(torch.cat([qa_lstm, qa_attend, qa_lstm * qa_attend, qa_lstm-qa_attend], dim=-1))

        return doc_encode.view(batch_size, -1, self.seq_length, self.hidden_size*2), qa_encode, qa_lstm, \
               doc_lstm.view(batch_size, -1, self.seq_length, self.hidden_size*2)

    def _create_mask(self, x1_len, x2_len, batch_size):
        """
        :param x1_len: see forward
        :param x2_len: see forward
        :return: shape: [batch_size, docset_size*seq_length, seq_length]
        """
        x1_len = x1_len.view(-1)
        x2_len = x2_len.view(-1)

        x1_mask = self._length_to_mask(x1_len, self.seq_length)
        x2_mask = self._length_to_mask(x2_len, self.seq_length)

        # shape: [batch_size, docset_size*seq_length, 1]
        x1_mask = x1_mask.view(batch_size, -1).view(batch_size, 1, -1).transpose(-2, -1)
        # shape: [batch_size, 1, seq_length]
        x2_mask = x2_mask.view(batch_size, -1).view(batch_size, 1, -1)

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

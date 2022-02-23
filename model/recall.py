# encoding:utf8
# Recall difficulty

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encode import LSTM
from .attention.single import Attention


class RecallModule(nn.Module):
    def __init__(self, batch_size, docset_size, seq_length, embedding_size, dropout):
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

        self.batch_size, self.docset_size, self.seq_length, self.embedding_size = \
            batch_size, docset_size, seq_length, embedding_size

        self.dropout = nn.Dropout(p=dropout)

        self.gating_layer = nn.Sequential(
            nn.Linear(embedding_size, 1),
            nn.ReLU()
        )

        self.fusion_layer = nn.Bilinear(embedding_size, embedding_size, 1)

        self.softmax = nn.Softmax(dim=-1)

        self.match_layer = nn.Sequential(
            nn.Linear(embedding_size * 2, embedding_size),
            nn.ReLU()
        )

    def forward(self, qa, x2, qa_len, x2_len):
        """
        :param qa: shape: [batch_size, seq_length, embedding_size]
        :param x2: shape: [batch_size, docset_size, seq_length, embedding_size]
        :param qa_len: shape: [batch_size,]
        :param x2_len: shape: [batch_size, docset_size]
        :return: shape: [batch_size, embedding_size*4]
        """
        # shape: [batch_size, docset_size]
        qa_len = qa_len.unsqueeze(1).repeat(1, self.docset_size)
        # shape: [batch_size * docset_size, seq_length, seq_length]
        mask = self._create_mask(qa_len, x2_len)
        #
        # # shape: [batch_size*docset_size, seq_length, embedding_size]
        x1 = qa.repeat(1, self.docset_size, 1).view(-1, self.seq_length, self.embedding_size)
        # # shape: [batch_size*docset_size, seq_length, embedding_size]
        x2 = x2.view(-1, self.seq_length, self.embedding_size)

        # shape: [batch_size*docset_size, seq_length, embedding_size]
        qa_attend, _ = self.attention(query=x1,
                                      key=x2,
                                      value=x2,
                                      mask=mask,
                                      dropout=self.dropout)
        qa_attend = qa_attend.view(self.batch_size, self.docset_size, self.seq_length, -1)
        x1 = x1.view(self.batch_size, self.docset_size, self.seq_length, -1)

        x1_cat = torch.cat([x1 - qa_attend, x1 * qa_attend], dim=-1)
        # batch_size, docset_size, seq_length, embedding_size
        word_match = self.match_layer(x1_cat)

        word_match_mean = torch.mean(word_match, dim=2)
        word_match_max = torch.max(word_match, dim=2)[0]

        # batch_size, docset_size, emb * 2
        mathc_h = torch.cat([word_match_mean, word_match_max], dim=-1)

        qa_mask = self._length_to_mask(qa_len.view(-1), self.seq_length).view(self.batch_size, self.docset_size, -1)
        doc_len = x2_len.view(-1)
        doc_mask = self._length_to_mask(doc_len, self.seq_length).view(self.batch_size, -1, self.seq_length)
        #
        # # shape: [batch_size, seq_length]
        x1_gates = self.gating_layer(x1).squeeze()
        x1_gates = x1_gates.masked_fill(qa_mask == 0, -1e9)
        x1_gates = self.softmax(x1_gates)
        x1_gated = x1 * x1_gates.unsqueeze(3)

        x1_score = torch.sum(x1_gated, dim=2)
        #
        # # shape: [batch_size, docset_size, seq_length]
        x2 = x2.view(self.batch_size, -1, self.seq_length, self.embedding_size)
        doc_gates = self.gating_layer(x2).squeeze()
        doc_gates = doc_gates.masked_fill(doc_mask == 0, -1e9)
        doc_gates = self.softmax(doc_gates)
        doc_gated = x2 * doc_gates.unsqueeze(3)
        #
        doc_score = torch.sum(doc_gated, dim=2)

        # # shape: [batch_size, docset_size, 1]
        sentence_match_score = self.fusion_layer(x1_score, doc_score)
        h = mathc_h * sentence_match_score

        output = self.fusion_layer2(h.view(self.batch_size, -1))

        return output, torch.zeros(3).float(), sentence_match_score

    def _create_mask(self, x1_len, x2_len):
        """
        :param x1_len: see forward
        :param x2_len: see forward
        :return: shape: [batch_size, seq_length, docset_size*seq_length]
        """
        x1_len = x1_len.view(-1)
        x2_len = x2_len.view(-1)

        batch_size = x2_len.size(0)

        x1_mask = self._length_to_mask(x1_len, self.seq_length)
        x2_mask = self._length_to_mask(x2_len, self.seq_length)

        # shape: [batch_size, seq_length, 1]
        x1_mask = x1_mask.view(batch_size, 1, -1).transpose(-2, -1)
        # shape: [batch_size, 1, docset_size*seq_length]
        x2_mask = x2_mask.view(batch_size, -1).view(batch_size, 1, -1)

        return x1_mask * x2_mask

    def _create_mask_for_self(self, doc_lens):
        """
        :param doc_lens: [batch_size, doc_size]
        :return: shape: [batch_size, docset_size*seq_length, docset_size*seq_length]
        """
        lens = doc_lens.view(-1)

        # shape: [batch_size*doc_size, seq_length]
        mask = self._length_to_mask(lens, self.seq_length)

        # shape: [batch_size, 1, doc_size*seq_length]
        mask = mask.view(self.batch_size, -1).view(self.batch_size, 1, -1)

        return mask.transpose(-2, -1) * mask

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




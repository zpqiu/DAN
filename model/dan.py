# encoding:utf-8

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

import model.embedding.token as token
from .confusion import ConfusionModule
from .recall import RecallModule
from .encode import Encoder


class Model(nn.Module):
    def __init__(self, vocab_size, config):
        super().__init__()

        self.embed = token.create_embedding_layer(weight_path=config["data"]["embedding"],
                                                  vocab_size=vocab_size,
                                                  embed_size=config["feature"]["embed_size"])

        self.encode = Encoder(batch_size=config["train"]["batch_size"],
                              seq_length=config["feature"]["seq_length"],
                              embedding_size=config["feature"]["embed_size"],
                              hidden_size=config["feature"]["encode_h_size"],
                              dropout=config["model"]["encode_dropout"])

        self.confusion_attend = ConfusionModule(batch_size=config["train"]["batch_size"],
                                          docset_size=1,
                                          seq_length=config["feature"]["seq_length"],
                                          encode_size=config["feature"]["embed_size"],
                                          dropout=config["model"]["encode_dropout"])

        self.recall_attend = RecallModule(batch_size=config["train"]["batch_size"],
                                                docset_size=config["feature"]["docset_size"],
                                                seq_length=config["feature"]["seq_length"],
                                                embedding_size=config["feature"]["embed_size"],
                                                dropout=config["model"]["encode_dropout"])

        self.batch_size, self.seq_length, self.embedding_size = \
            config["train"]["batch_size"], config["feature"]["seq_length"], config["feature"]["embed_size"]

        self.student = nn.Parameter(torch.randn(config["feature"]["student_n"]))
        self.student_n = config["feature"]["student_n"]

        self.h1 = nn.Sequential(nn.Linear(config["feature"]["embed_size"] * 2, 1), nn.Sigmoid())
        self.h2 = nn.Sequential(nn.Linear(config["feature"]["embed_size"] * 2, 1), nn.Sigmoid())

        self.alpha = nn.Sequential(nn.Linear(config["feature"]["embed_size"] * 2, 1), nn.Sigmoid())

    def forward(self, qa, other_qa, doc_set, other_doc_set):
        # Embed.
        # shape: [batch_size, seq_length, embedding_size]
        qa_embed = self.embed(qa[0])
        # shape: [batch_size, 4, seq_length, embedding_size]
        other_qa_embed = self.embed(other_qa[0])
        # shape: [batch_size, docset_size, seq_length, embedding_size]
        doc_set_embed = self.embed(doc_set[0])
        # shape: [batch_size, 4, docset_size, seq_length, embedding_size]
        other_doc_set_embed = self.embed(other_doc_set[0])

        # Encode
        # shape: [batch_size, docset_size, seq_length, encode_size]
        doc_encode, qa_encode, qa_lstm, doc_lstm = self.encode(qa_embed, qa[1], doc_set_embed, doc_set[1])
        qa_encode = qa_encode.squeeze()

        other_qa_embed = other_qa_embed.view(self.batch_size * 4, self.seq_length, self.embedding_size)
        other_doc_set_embed = other_doc_set_embed.view(self.batch_size * 4, -1, self.seq_length, self.embedding_size)
        other_qa_len = other_qa[1].view(-1)
        other_doc_set_lens = other_doc_set[1].view(self.batch_size * 4, -1)

        # shape: [batch_size*4, docset_size, seq_length, encode_size]
        other_doc_encode, other_qa_encode, _, _ = self.encode(other_qa_embed, other_qa_len, other_doc_set_embed,
                                                              other_doc_set_lens)
        # embedding_size == encode_size
        other_doc_encode = other_doc_encode.view(self.batch_size, 4, -1, self.seq_length, self.embedding_size)

        qa_encode = qa_encode.view(self.batch_size, -1, self.seq_length, self.embedding_size)
        other_qa_encode = other_qa_encode.view(self.batch_size, 4, -1, self.seq_length, self.embedding_size)
        qa_lens = qa[1].view(self.batch_size, 1)
        other_qa_lens = other_qa[1].view(self.batch_size, 4, 1)
        # shape: [batch_size, encode_size*2]

        confusion_hidden, _ = self.confusion_attend(qa_encode, other_qa_encode, qa_lens, other_qa_lens)
        recall_hidden, gates, _ = self.recall_attend(qa_lstm.squeeze(),
                                                 doc_encode,
                                                 qa[1],
                                                 doc_set[1])

        confusion_score = self.h1(confusion_hidden)
        recall_score = self.h2(recall_hidden)

        qa_lstm = qa_lstm.squeeze()
        qa_max = torch.max(qa_lstm, dim=1)[0]
        qa_mean = torch.mean(qa_lstm, dim=1)
        alpha = self.alpha(torch.cat([qa_max, qa_mean], dim=-1))

        output = alpha * confusion_score + (1.0-alpha) * recall_score
        output = output.squeeze()

        return output, gates, alpha, confusion_score, recall_score

    def predict_score(self, difficulties):
        student = self.student.unsqueeze(0).repeat(self.batch_size, 1)
        diffs = difficulties.unsqueeze(1).repeat(1, self.student_n)

        # shape: [batch_size, student_n]
        # score = torch.exp(student-diffs) / (1.0 + torch.exp(student-diffs))
        score = student-diffs
        score = score.view(-1, 1)
        zero_score = torch.zeros(self.batch_size * self.student_n, dtype=score.dtype, device=score.device).view(-1, 1)
        return torch.cat([zero_score, score], dim=-1)

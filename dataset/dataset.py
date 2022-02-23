from torch.utils.data import Dataset
import tqdm
import torch
import random
import json


class ModelDataset(Dataset):
    def __init__(self, corpus_path, score_path, vocab, q_seq_len, doc_seq_len, train_ratio=1.0, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.q_seq_len = q_seq_len
        self.doc_seq_len = doc_seq_len

        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line.strip()
                              for line in tqdm.tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                if train_ratio < 1.0:
                    self.lines = self.lines[:int(train_ratio * len(self.lines))]
                self.corpus_lines = len(self.lines)

        self.score_dict = dict()
        lines = open(score_path, "r").readlines()
        for line in lines:
            r = line.strip().split(",")
            self.score_dict[r[0]] = [int(float(x)) for x in r[1:]]

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        qa, other_qa, doc_set, other_doc_set, difficulty, score = self.parse_line(item)

        # shape: [q_seq_len, ]
        qa_seq, qa_len = self.vocab.to_seq(qa, seq_len=self.q_seq_len, with_len=True)

        # other_qa_seq shape: [4, q_seq_len]
        # other_qa_len shape: [4, ]
        other_qa_seq, other_qa_len = [], []
        for qa in other_qa:
            seq, q_len = self.vocab.to_seq(qa, seq_len=self.q_seq_len, with_len=True)
            other_qa_seq.append(seq)
            other_qa_len.append(q_len)

        # doc_set_seqs shape: [doc_set_size, doc_seq_len]
        # doc_set_lens shape: [doc_set_size, ]
        doc_set_seqs, doc_set_lens = [], []
        for doc in doc_set:
            doc_seq, doc_len = self.vocab.to_seq(doc, seq_len=self.doc_seq_len, with_len=True)
            doc_set_seqs.append(doc_seq)
            doc_set_lens.append(doc_len)

        # other_doc_set_seqs shape: [4, doc_set_size, doc_seq_len]
        # other_doc_set_lens shape: [4, doc_set_size, ]
        other_doc_set_seqs, other_doc_set_lens = [], []

        for docs in other_doc_set:
            each_option_doc_seqs, each_option_doc_lens = [], []
            for doc in docs:
                doc_seq, doc_len = self.vocab.to_seq(doc, seq_len=self.doc_seq_len, with_len=True)
                each_option_doc_seqs.append(doc_seq)
                each_option_doc_lens.append(doc_len)
            other_doc_set_seqs.append(each_option_doc_seqs)
            other_doc_set_lens.append(each_option_doc_lens)

        output = {"qa": qa_seq,
                  "qa_len": qa_len,
                  "other_qa": other_qa_seq,
                  "other_qa_len": other_qa_len,
                  "doc_set": doc_set_seqs,
                  "doc_set_len": doc_set_lens,
                  "other_doc_set": other_doc_set_seqs,
                  "other_doc_set_len": other_doc_set_lens,
                  "difficulty": difficulty,
                  "score": score
                  }

        return {key: torch.tensor(value) for key, value in output.items()}

    def parse_line(self, item):
        line = self.lines[item]
        j = json.loads(line)

        qa = j["qa"]
        other_qa = j["other_qa"]
        doc_set = j["doc_set"]
        other_doc_set = j["other_doc_set"]

        difficulty = j["difficulty"]

        score = self.score_dict[j["qid"]]

        return qa, other_qa, doc_set, other_doc_set, difficulty, score

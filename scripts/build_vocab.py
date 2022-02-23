# encoding: utf-8
"""
Author: zhaopeng qiu
Date: 10 Feb, 2019
      update at Apr. 17, 2019
"""


import os
import argparse
import numpy as np
from dataset.vocab import WordVocab


def build_weights(vocab, embedding_file, weights_output_file, embed_size):
    # Load 预训练的embedding
    lines = open(embedding_file, "r", encoding="utf8").readlines()[1:]
    emb_dict = dict()
    for line in lines:
        row = line.split()
        embedding = [float(w) for w in row[1:]]
        emb_dict[row[0]] = np.array(embedding)

    # build embedding weights for model
    weights_matrix = np.zeros((len(vocab), embed_size))
    words_found = 0

    for i, word in enumerate(vocab.itos):
        try:
            weights_matrix[i] = emb_dict[word]
            words_found += 1
        except KeyError:
            weights_matrix[i] = np.random.normal(size=(embed_size,))
    np.save(weights_output_file, weights_matrix)


def build():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-e", "--encoding", type=str, default="utf-8")
    parser.add_argument("-m", "--min_freq", type=int, default=1)
    parser.add_argument("-eb", "--embedding_file", type=str, default=None)
    parser.add_argument("-wo", "--weights_output_file", type=str, default=None)
    parser.add_argument("-ez", "--embed_size", type=int, default=200)
    args = parser.parse_args()

    output_file_name = ".".join(os.path.basename(args.corpus_path).split(".")[:-1]) + ".vocab"
    output_path = os.path.join(os.path.dirname(os.path.abspath(args.corpus_path)), output_file_name)

    with open(args.corpus_path, "r", encoding=args.encoding) as f:
        vocab = WordVocab(f, max_size=args.vocab_size, min_freq=args.min_freq)

    build_weights(vocab, args.embedding_file, args.weights_output_file, args.embed_size)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(output_path)


if __name__ == "__main__":
    build()

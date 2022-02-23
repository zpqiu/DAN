# encoding: utf-8
"""
Author: zhaopeng qiu
Date: 12 Feb, 2019
      update at Apr. 17, 2019
"""
import toml
import argparse
import logging
import random

import torch
from torch.utils.data import DataLoader

from model.dan import Model
from trainer.train import Trainer
from dataset.dataset import ModelDataset
from dataset.vocab import WordVocab
from tester.test import Tester
from utils import util


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-cf", "--config", type=str, required=True, help="config file path")
    parser.add_argument("--mode", type=int, default=0, help="0 for training, 1 for testing")
    parser.add_argument("--epoch_for_test", type=int, default=0, help="the test model's epoch")
    parser.add_argument("--r", type=float, default=1.0, help="the train data ratio")

    args = parser.parse_args()

    config_str = open(args.config, "r", encoding="utf-8").read()
    config = toml.loads(config_str)

    util.set_logger(config["train"]["model_output_path"] + ".log")

    logging.info("Loading Vocab, {0}".format(config["data"]["vocab_path"]))
    vocab = WordVocab.load_vocab(config["data"]["vocab_path"])
    logging.info("Vocab Size: {0}".format(len(vocab)))

    logging.info("Loading Train Dataset, {0}".format(config["data"]["train_dataset"]))
    seq_len, batch_size = config["feature"]["seq_length"], config["train"]["batch_size"]
    train_dataset = ModelDataset(config["data"]["train_dataset"], config["data"]["score_path"], vocab, q_seq_len=seq_len, doc_seq_len=seq_len,
                                 corpus_lines=None, on_memory=True, train_ratio=args.r)

    logging.info("Loading Test Dataset, {0}".format(config["data"]["test_dataset"]))
    test_dataset = ModelDataset(config["data"]["test_dataset"], config["data"]["score_path"], vocab, q_seq_len=seq_len, doc_seq_len=seq_len,
                                on_memory=True) if config["data"]["test_dataset"] != "" else None

    logging.info("Loading Dev Dataset, {0}".format(config["data"]["dev_dataset"]))
    dev_dataset = ModelDataset(config["data"]["dev_dataset"], config["data"]["score_path"], vocab, q_seq_len=seq_len, doc_seq_len=seq_len,
                                on_memory=True) if config["data"]["dev_dataset"] != "" else None

    logging.info("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1,
                                   shuffle=True, drop_last=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1,
                                  drop_last=True) if test_dataset is not None else None
    dev_data_loader = DataLoader(dev_dataset, batch_size=batch_size, num_workers=1,
                                  drop_last=True) if dev_dataset is not None else None

    logging.info("Building model")
    model = Model(len(vocab), config=config)

    if args.mode not in [0, 1]:
        logging.warning("Mode should be 0 or 1.")
        return

    if args.mode == 0:
        logging.info("Creating Trainer")
        trainer = Trainer(model, len(vocab), config=config,
                          train_dataloader=train_data_loader, test_dataloader=dev_data_loader)

        logging.info("Training Start")
        for epoch in range(config["train"]["epoch"]):
            trainer.train(epoch)
            trainer.save(epoch)

            if dev_data_loader is not None:
                trainer.test(epoch)
    else:
        logging.info("Creating Tester")
        tester = Tester(model, config=config, test_dataloader=test_data_loader)
        predict_result_path = config["test"]["predict_result_path"]
        test(tester, args, predict_result_path)


def test(tester, args, predict_result_path):
    tester.load(args.epoch_for_test)
    predict_list, real_list, alpha_list, mem_list, rea_list = tester.predict()

    rmse, mae, scc, kcc = tester.evaluate(predict_list, real_list)
    logging.info("RMSE: {0}, MAE: {1}, SCC: {2}, PCC:{3}".format(rmse, mae, scc, kcc))

    with open(predict_result_path, "w", encoding="utf8") as fw:
        for p, r, a, m, e in zip(predict_list, real_list, alpha_list, mem_list, rea_list):
            fw.write("{0}\t{1}\t{2}\t{3}\t{4}\n".format(p, r, a, m, e))


if __name__ == '__main__':
    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)
    train()

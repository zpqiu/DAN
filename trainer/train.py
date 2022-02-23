import torch
import torch.nn as nn
from utils.optim import AdamW
from torch.utils.data import DataLoader

import tqdm
import logging
from scipy.stats import pearsonr
import numpy as np


class Trainer:
    def __init__(self, model, vocab_size: int, config: dict,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None):
        """
        :param model: model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param config: config dict
        """
        self.config = config

        # Setup cuda device for model training
        cuda_condition = torch.cuda.is_available() and config["train"]["with_cuda"]
        self.device = torch.device(("cuda:"+config["train"]["gpu"]) if cuda_condition else "cpu")
        self.model = model.to(self.device)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = AdamW(self.model.parameters(),
                          lr=config["train"]["lr"],
                          weight_decay=config["train"]["weight_decay"])

        # Using MSE Loss function for predicting the difficulty
        self.criterion = nn.MSELoss()
        self.score_criterion = nn.CrossEntropyLoss()

        self.log_freq = config["log"]["log_freq"]

        self.lr, self.warmup, self.epoch_count = \
            config["train"]["lr"], config["train"]["warmup"], config["train"]["epoch"]

        logging.info("Total Parameters: {0}".format(sum([p.nelement() for p in self.model.parameters()])))

    def train(self, epoch):
        self.model.train()
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.model.eval()
        with torch.no_grad():
            self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """
        loop over the data_loader for training or testing
        if on train status, backward operation is activated
        and also auto save the model every epoch

        :param epoch: current epoch index
        :param data_loader: torch.utils.data.DataLoader for iteration
        :param train: boolean value of is train or test
        :return: None
        """
        str_code = "train" if train else "test"

        steps_one_epoch = len(data_loader)
        train_steps = steps_one_epoch * self.epoch_count

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        avg_mse_loss = 0.0
        total_loss = 0.0
        total_mse_loss = 0.0
        predict_list = []
        real_list = []

        for i, data in data_iter:
            # 0. batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            # 1. forward the qa and doc_set
            predict_difficulty, gates, _, _, _ = self.model.forward((data["qa"], data["qa_len"]),
                                                    (data["other_qa"], data["other_qa_len"]),
                                                    (data["doc_set"], data["doc_set_len"]),
                                                    (data["other_doc_set"], data["other_doc_set_len"]))

            # 2. MSE loss
            mse_loss = self.criterion(predict_difficulty, data["difficulty"])
            gate_term = torch.clamp(torch.mean(gates) - 0.7, min=0.0, max=1.0)
            mse_loss = mse_loss + 0.01 * gate_term

            loss = mse_loss # + 0.1 * score_loss

            # 3. backward and optimization only in train
            if train:
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()
            else:
                predict_list += predict_difficulty.data.cpu().numpy().tolist()
                real_list += data["difficulty"].data.cpu().numpy().tolist()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": loss,
                "avg_mse_loss": mse_loss,
            }

            avg_loss += loss.data.cpu().numpy()
            avg_mse_loss += mse_loss.data.cpu().numpy()

            if (i+1) % self.log_freq == 0:
                post_fix["avg_mse_loss"] = avg_mse_loss / self.log_freq
                post_fix["avg_loss"] = avg_loss / self.log_freq
                data_iter.write(str(post_fix))
                total_loss += avg_loss
                total_mse_loss += avg_mse_loss
                avg_loss = 0.0
                avg_mse_loss = 0.0

        logging.info("EP{0}_{1}, avg_mse_loss={2}, avg_loss={3}".format(epoch,
                                                                        str_code,
                                                                        total_mse_loss / len(data_iter),
                                                                        total_loss / len(data_iter)))

        if not train:
            rmse, doa, pcc = self.evaluate(predict_list, real_list)
            logging.info("RMSE: {0}, DOA: {1}, PCC: {2}".format(rmse, doa, pcc))

    def evaluate(self, predict, real):
        """
        多个metrics
        :param predict:
        :param real:
        :return: Tuple(float, float, float), 1. RMSE, 2. degree of agreement, 3. pearson
        """
        predict = np.array(predict)
        real = np.array(real)

        rmse = np.sqrt(((predict-real)**2).mean())
        doa = 0
        N = len(predict)
        for x, y in zip(real.argsort(), predict.argsort()):
            doa += (N-max(x, y))
        doa = doa/(N*(N+1))*2
        pcc = pearsonr(real, predict)

        return rmse, doa, pcc

    def save(self, epoch):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :return: final_output_path
        """
        file_path = self.config["train"]["model_output_path"]
        output_path = file_path + ".ep%d" % epoch
        self.model.cpu()
        torch.save(self.model.state_dict(), output_path)
        self.model.to(self.device)
        logging.info("EP:{0} Model Saved on: {1}".format(epoch, output_path))
        return output_path

    def creat_mask(self, score):
        sc = score.view(-1)
        mask = sc.ge(0)
        logits_mask = mask.view(-1, 1).repeat(1, 2)
        return mask, logits_mask

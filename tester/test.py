# encoding: utf-8
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import tqdm
import numpy as np

from sklearn.metrics import cohen_kappa_score
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from scipy.stats import kendalltau


class Tester:
    def __init__(self, model, config: dict, test_dataloader: DataLoader = None):
        """
        :param model: model which you want to train
        :param test_dataloader: test dataset data loader [can be None]
        :param config: config dict
        """
        self.config = config

        # Setup cuda device for model training
        cuda_condition = torch.cuda.is_available() and config["train"]["with_cuda"]
        self.device = torch.device(("cuda:"+config["train"]["gpu"]) if cuda_condition else "cpu")

        self.model = model.to(self.device)

        # Setting the train and test data loader
        self.test_data = test_dataloader

        # Using MSE Loss function for predicting the difficulty
        self.criterion = nn.MSELoss()

        self.log_freq = config["log"]["log_freq"]

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def predict(self):
        """
        predict test
        :return: Tuple(list[float], list[float]), predict list and real list
        """
        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(self.test_data),
                              desc="EP_test:0",
                              total=len(self.test_data),
                              bar_format="{l_bar}{r_bar}")

        self.model.eval()

        predict_list = []
        real_list = []
        alpha_list, mem_list, rea_list = [], [], []

        with torch.no_grad():
            for i, data in data_iter:
                # 0. batch_data will be sent into the device(GPU or cpu)
                data = {key: value.to(self.device) for key, value in data.items()}

                # 1. forward the qa and doc_set
                predict_difficulty,  _, alpha, mem, rea = self.model.forward((data["qa"], data["qa_len"]),
                                                    (data["other_qa"], data["other_qa_len"]),
                                                    (data["doc_set"], data["doc_set_len"]),
                                                    (data["other_doc_set"], data["other_doc_set_len"]))

                # 2. append
                predict_list += predict_difficulty.data.cpu().numpy().tolist()
                real_list += data["difficulty"].data.cpu().numpy().tolist()
                alpha_list += alpha.data.cpu().numpy().tolist()
                mem_list += mem.data.cpu().numpy().tolist()
                rea_list += rea.data.cpu().numpy().tolist()

        return predict_list, real_list, alpha_list, mem_list, rea_list

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
        mae = np.abs(predict-real).mean()
        scc = spearmanr(real, predict)
        kcc = kendalltau(real, predict)

        return rmse, mae, scc, kcc

    def load(self, epoch):
        """
        Saving the current model on file_path

        :param epoch: current epoch number
        :return: final_output_path
        """
        file_path = self.config["train"]["model_output_path"]
        output_path = file_path + ".ep%d" % epoch

        pretrained_model = torch.load(output_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(pretrained_model)
        self.model.to(self.device)

        print("EP:%d Model Loaded" % epoch)
        return output_path

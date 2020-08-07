from pandas_datareader import data as wb
from torch.utils.data import Dataset, DataLoader
import torch 
import numpy as np

class FinancialDataLoader(Dataset):
    COLUMNS_TO_DROP = ['Open', 'Volume', 'High', 'Low', 'Adj Close']
    COLUMNS_TO_NORMALIZE = ['Volume']

    def __init__(self, ticker):
        self.dataset = self.query_data(ticker=ticker)
        self.drop_unnecessary_columns()

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset.values[idx], dtype=torch.float32)

    def query_data(self, ticker):
        return wb.DataReader(ticker, data_source='yahoo')

    def drop_unnecessary_columns(self):
        self.dataset.drop(self.COLUMNS_TO_DROP, axis=1, inplace=True)

    def as_tensor_list(self):
        storage = torch.zeros(len(self), dtype=torch.float32)
        for i in range(len(storage)):
            storage[i] = self[i]
        return storage

    def normalize_columns(self):
        pass

    def get_dataset(self):
        return self.dataset


class FinancialDataIterator:
    BATCH_SIZE = 10
    TRAIN_PERCENTAGE = 0.7
    TEST_PERCENTAGE = 1 - TRAIN_PERCENTAGE

    def __init__(self, data_tensor, tau):
        self.data = data_tensor
        self.tau = tau

    def partition_data(self, is_train):
        data_length = len(self.data)
        if is_train:
            train_len = round(self.TRAIN_PERCENTAGE * data_length)
            return self.data[0:train_len]
        test_len = round(self.TEST_PERCENTAGE * data_length)
        return self.data[-test_len:]

    def prepare_data(self, torch_data):
        T = len(torch_data)
        features = torch.zeros((T-self.tau, self.tau))
        for i in range(self.tau):
            features[:, i] = torch_data[i: T-self.tau+i]
            labels = torch.reshape(torch_data[self.tau:], (-1, 1))

        return features, labels

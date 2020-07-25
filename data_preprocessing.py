import pandas as pd
from pandas_datareader import data as wb
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch 
from d2l import mxnet as d2l

class FinancialDataLoader(Dataset):
    COLUMNS_TO_DROP = ['Adj Close']
    COLUMNS_TO_NORMALIZE = ['Volume']

    def __init__(self, ticker):
        self.dataset = self.query_data(ticker=ticker)
        self.drop_unnecessary_columns()

    def __len__(self):
        return len(self.dataset.index)

    def __getitem__(self, idx):
        return torch.tensor(self.dataset.values[idx])

    def query_data(self, ticker):
        return wb.DataReader(ticker, data_source='yahoo')

    def drop_unnecessary_columns(self):
        self.dataset.drop(self.COLUMNS_TO_DROP, axis=1, inplace=True)

    def as_tensor(self):
        return torch.tensor(self.dataset.values)

    def normalize_columns(self):
        pass

    def get_dataset(self):
        return self.dataset


FData = FinancialDataLoader('NCLH')
data_iter = DataLoader(FData, batch_size=10)

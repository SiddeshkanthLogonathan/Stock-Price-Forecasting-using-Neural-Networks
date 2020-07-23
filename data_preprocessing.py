import pandas as pd
from pandas_datareader import data as wb

class FinancialDataLoader:
    COLUMNS_TO_DROP = ['Adj Close']
    COLUMNS_TO_NORMALIZE = ['Volume']

    def __init__(self, ticker):
        self.dataset = self.query_data(ticker=ticker)
        self.drop_unnecessary_columns()

    def query_data(self, ticker):
        return wb.DataReader(ticker, data_source='yahoo')

    def drop_unnecessary_columns(self):
        self.dataset.drop(self.COLUMNS_TO_DROP, axis=1, inplace=True)

    def as_numpy_array(self):
        return self.dataset.to_numpy()

    def normalize_columns(self):
        pass

    def get_dataset(self):
        return self.dataset



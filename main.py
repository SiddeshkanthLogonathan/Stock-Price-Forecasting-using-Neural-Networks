from data_preprocessing import FinancialDataLoader, FinancialDataIterator
from data_visualization import FinancialDataBuilder
import argparse as arg
import sys

def process_data(ticker):
    FData = FinancialDataLoader(ticker)
    data_iter = FinancialDataIterator(FData)
    train_iter = data_iter.partition_data(is_train=True)
    test_iter = data_iter.partition_data(is_train=False)
    return train_iter, test_iter

def main():
    ERROR_MESSAGE = 'Error: Please enter a ticker symbol. (ex: FB)'

    parser = arg.ArgumentParser()
    parser.add_argument('--ticker', nargs='?', default=ERROR_MESSAGE)
    args = parser.parse_args()

    ticker = args.ticker
    curated_data = process_data(ticker=ticker)
    train_data = curated_data[0]
    test_data = curated_data[1]
    


if __name__ == '__main__':
    main()
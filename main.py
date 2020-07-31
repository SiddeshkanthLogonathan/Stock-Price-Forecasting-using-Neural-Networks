from data_preprocessing import FinancialDataLoader, FinancialDataIterator
from data_visualization import FinancialDataBuilder, FinancialDataVisualizer
import argparse as arg
import sys

def process_data(ticker):
    print('Processing Data...')
    FData = FinancialDataLoader(ticker)
    data_iter = FinancialDataIterator(FData, tau=2)
    train_iter = data_iter.partition_data(is_train=True)
    test_iter = data_iter.partition_data(is_train=False)
    return FData, train_iter, test_iter

def train_model():
    print('Training Model...')
    pass 

def visualize_data(ticker, curated_data):
    print('Visualizing Data...')
    FData_object = curated_data[0]
    train_data = curated_data[1]
    test_data = curated_data[2]

    data_visualizer = FinancialDataVisualizer(train_data, test_data, FData_object)
    data_visualizer.visualize(title=ticker)

def main():
    ERROR_MESSAGE = 'Error: Please enter a ticker symbol. (ex: --ticker FB)'

    parser = arg.ArgumentParser()
    parser.add_argument('--ticker', nargs='?', default=None)
    args = parser.parse_args()

    ticker = args.ticker
    if ticker == None:
        sys.stdout.write(ERROR_MESSAGE + '\n')
        sys.exit(25)
    curated_data = process_data(ticker=ticker)



if __name__ == '__main__':
    main()
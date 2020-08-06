from data_preprocessing import FinancialDataLoader, FinancialDataIterator
from data_visualization import FinancialDataBuilder, FinancialDataVisualizer
import model
import training
import argparse as arg
import sys
from d2l import torch as d2l
import torch
import torch.nn as nn

def process_data(data_iterator, data_tensor):
    train_data = data_iterator.partition_data(is_train=True)
    test_data = data_iterator.partition_data(is_train=False)

    return train_data, test_data

def main():
    ERROR_MESSAGE = 'Error: Please enter a ticker symbol. (ex: --ticker FB)'
    TAU, BATCH_SIZE = 4, 128

    parser = arg.ArgumentParser()
    parser.add_argument('--ticker', nargs='?', default=None)
    args = parser.parse_args()

    ticker = args.ticker
    if ticker == None:
        sys.stdout.write(ERROR_MESSAGE + '\n')
        sys.exit(25)

    print('Processing Data...')
    data = FinancialDataLoader(ticker=ticker)
    data_tensor = data.as_tensor_list()
    data_iterator = FinancialDataIterator(data_tensor=data_tensor, tau=TAU)

    train_data, test_data = process_data(data_iterator=data_iterator, data_tensor=data_tensor)
    train_feature, train_label = data_iterator.prepare_data(torch_data=train_data)
    test_feature, test_label = data_iterator.prepare_data(torch_data=test_data)

    print('Preparing Model...')
    train_iter = d2l.load_array((train_feature, train_label), BATCH_SIZE, is_train=True)
    test_iter = d2l.load_array((test_feature, test_label), BATCH_SIZE, is_train=False)
    net = model.get_net(input_size=TAU)

    print('Training...')
    loss = nn.MSELoss()
    num_epochs = 1000
    lr = 0.01
    training.train_net(net, train_iter, loss, num_epochs, lr)

    X = net(train_feature).detach()
    y = net(test_feature).detach() 

    last_four_days = test_feature[-1]
    price_of_tommorow = net(last_four_days)
    print('Next day closing price: ' + str(price_of_tommorow.detach().numpy()))

    print('Visualizing Data...')
    data_visualizer = FinancialDataVisualizer(df_index=data.get_dataset().index, data=data, model_data=(X, y), tau=TAU)
    data_visualizer.visualize(title=ticker)

if __name__ == '__main__':
    main()
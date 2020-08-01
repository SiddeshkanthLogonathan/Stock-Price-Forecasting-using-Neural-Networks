from data_preprocessing import FinancialDataLoader, FinancialDataIterator
from data_visualization import FinancialDataBuilder, FinancialDataVisualizer
import model
import training
import argparse as arg
import sys
from d2l import torch as d2l
import torch
import torch.nn as nn

def main():
    ERROR_MESSAGE = 'Error: Please enter a ticker symbol. (ex: --ticker FB)'

    parser = arg.ArgumentParser()
    parser.add_argument('--ticker', nargs='?', default=None)
    args = parser.parse_args()

    ticker = args.ticker
    if ticker == None:
        sys.stdout.write(ERROR_MESSAGE + '\n')
        sys.exit(25)

    ## ========================================================================= 

    print('Processing Data...')
    data = FinancialDataLoader(ticker=ticker)
    data_tensor = data.as_tensor_list()

    tau = 4
    data_i = FinancialDataIterator(data_tensor, tau=tau)
    train_data = data_i.partition_data(is_train=True)
    test_data = data_i.partition_data(is_train=False)

    train_feature, train_label = data_i.prepare_data(torch_data=train_data)
    test_feature, test_label = data_i.prepare_data(torch_data=test_data)

    ## ========================================================================= 

    print('Preparing Model...')
    batch_size = 128
    train_iter = d2l.load_array((train_feature, train_label), batch_size, is_train=True)
    test_iter = d2l.load_array((test_feature, test_label), batch_size, is_train=False)

    net = model.get_net(input_size=tau)

    ## ========================================================================= 

    print('Training...')
    loss = nn.MSELoss()
    num_epochs = 1000
    lr = 0.01
    training.train_net(net, train_iter, loss, num_epochs, lr)

    ## ========================================================================= 

    X = net(train_feature).detach()
    y = net(test_feature).detach() 

    ## ========================================================================= 
    print('Visualizing Data...')
    f_v = FinancialDataVisualizer(df_index=data.get_dataset().index, data=data, model_data=(X, y), tau=tau)
    f_v.visualize(title=ticker)


if __name__ == '__main__':
    main()
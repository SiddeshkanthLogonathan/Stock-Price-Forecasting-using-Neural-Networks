from data_preprocessing import FinancialDataLoader, FinancialDataIterator
from data_visualization import FinancialDataBuilder, FinancialDataVisualizer
import argparse as arg
import sys

def process_data(ticker, tau):
    print('Processing Data...')
    data = FinancialDataLoader('NCLH')
    data_tensor = data.as_tensor_list()

    data_i = FinancialDataIterator(data_tensor, tau=tau)
    train_data = data_i.partition_data(is_train=True)
    test_data = data_i.partition_data(is_train=False)

    train_feature, train_label = data_i.prepare_data(torch_data=train_data)
    test_feature, test_label = data_i.prepare_data(torch_data=test_data)

    return train_feature, train_label, test_feature, test_label

def prepare_model(curated_data, batch_size, tau):
    print('Preparing Model...')
    train_iter = d2l.load_array((curated_data[0], curated_data[1]), batch_size, is_train=True)
    test_iter = d2l.load_array((curated_data[2], curated_data[3]), batch_size, is_train=False)

    net = get_net(input_size=tau)
    return train_iter, test_iter, net

def train_model(net, train_iter, num_epochs, lr):
    print('Training...')
    loss = nn.MSELoss()
    train_net(net, train_iter, loss, num_epochs, lr)
    return net

def finalizing_build(net, train_feature, test_feature):
    print('Finalizing build...')
    X = net(train_feature).detach()
    y = net(test_feature).detach()

    return X, y

def visualize(df_index, data, model_data, tau, title)
    print('Visualizing data...')
    f_v = FinancialDataVisualizer(df_index=df_index, data=data, model_data=model_data, tau=tau)
    f_v.visualize(title)

def main():
    ERROR_MESSAGE = 'Error: Please enter a ticker symbol. (ex: --ticker FB)'

    parser = arg.ArgumentParser()
    parser.add_argument('--ticker', nargs='?', default=None)
    args = parser.parse_args()

    ticker = args.ticker
    if ticker == None:
        sys.stdout.write(ERROR_MESSAGE + '\n')
        sys.exit(25)
    curated_data = process_data(ticker, tau=4)
    



if __name__ == '__main__':
    main()
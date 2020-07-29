from d2l import torch as d2l
import torch
import torch.nn as nn
from data_preprocessing import FinancialDataLoader, FinancialDataIterator
from data_visualization import FinancialDataVisualizer

data = FinancialDataLoader('NCLH')
data_tensor = data.as_tensor_list()

T = len(data) 
tau = 2

data_i = FinancialDataIterator(data_tensor, tau=tau)
train_data = data_i.partition_data(is_train=True)
test_data = data_i.partition_data(is_train=False)

train_feature, train_label = data_i.prepare_data(torch_data=train_data)
test_feature, test_label = data_i.prepare_data(torch_data=test_data)

batch_size = 128

train_iter = d2l.load_array((train_feature, train_label), batch_size, is_train=True)
test_iter = d2l.load_array((test_feature, test_label), batch_size, is_train=False)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

# Vanilla MLP architecture
def get_net(input_size):
    net = nn.Sequential(nn.Linear(input_size, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

# Least mean squares loss
loss = nn.MSELoss()

def train_net(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(1, epochs + 1):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'    epoch {epoch}, loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net(input_size=tau)
# print(net)

# trained_data = torch.cat((data[0:tau], net(features)), dim=0)
# data_visualizer = FinancialDataVisualizer(trained_data.detach().numpy(), data)
# data_visualizer.visualize('NCLH')
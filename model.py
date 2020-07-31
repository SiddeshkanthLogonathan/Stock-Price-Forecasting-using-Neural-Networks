from d2l import torch as d2l
import torch
import torch.nn as nn
from data_preprocessing import FinancialDataLoader, FinancialDataIterator
from data_visualization import FinancialDataVisualizer, FinancialDataBuilder

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


def as_tensor_list(data):
    storage = torch.zeros(len(data), dtype=torch.float32)
    for i in range(len(storage)):
        storage[i] = data[i]
    return storage


train_net(net, train_iter, loss, 50, 0.01)
X = as_tensor_list(net(train_feature).detach())
y = as_tensor_list(net(test_feature).detach())



# print(X)
# print(y)


# data_visualizer = FinancialDataVisualizer(data, (X, y), tau)
# data_visualizer.visualize('NCLH')

d_b = FinancialDataVisualizer(initial_data=data.get_dataset(), model_data=(X, y), tau=tau)
d_v.visualize('NCLH')
# print(d_b.train_df)
# print(d_b.test_df)
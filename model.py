from data_preprocessing import FinancialDataLoader, FinancialDataIterator
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from d2l import mxnet as d2l


f_data = FinancialDataLoader(ticker='NCLH')
data = f_data.as_tensor()

# Move to data_preprocessing
Tau = 4 # using the last 4 days to predict the fifth day
T = len(f_data)
features = torch.zeros((T - Tau, Tau))

for i in range(Tau):
    features[:, i] = data[i:T-Tau+i, 0]

labels = torch.reshape(data[Tau:], (-1,1))

batch_size = 256

# by calling this, it has to give me batches that I'm expecting
data_loader = DataLoader(f_data, batch_size=10, shuffle=False) # this is the iterator

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

loss = nn.MSELoss()

def train_net(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(1, epochs + 1):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        print(f'epoch {epoch}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
# train_net(net, train_iter, loss, 10, 0.01)

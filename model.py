from d2l import torch as d2l
import torch
import torch.nn as nn
from data_preprocessing import FinancialDataLoader

data = FinancialDataLoader('NCLH')
# print(data.as_tensor_list())
# print(data[2:10])
data = data.as_tensor_list()

T = len(data)  # Generate a total of 1000 points
time = torch.arange(0, T, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))
# print(x[3:7])
tau = 4
# print(x[3])
# print(data[3:10])
features = torch.zeros((T-tau, tau))
for i in range(tau):
    features[:, i] = data[i: T-tau+i]
labels = d2l.reshape(data[tau:], (-1, 1))

batch_size, n_train = 16, 600
train_iter = d2l.load_array((features, labels), batch_size, is_train=True)

# Function for initializing the weights of net
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)

# Vanilla MLP architecture
def get_net():
    net = nn.Sequential(nn.Linear(4, 10),
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
        print(f'epoch {epoch}, '
              f'loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

net = get_net()
# print(net)
train_net(net, train_iter, loss, 100, 0.1)
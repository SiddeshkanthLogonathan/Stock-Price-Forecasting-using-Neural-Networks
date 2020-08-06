import torch
from d2l import torch as d2l

def train_net(net, train_iter, loss, epochs, lr):
    trainer = torch.optim.Adam(net.parameters(), lr)
    for epoch in range(1, epochs + 1):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            trainer.step()
        if epoch % 100 == 0 or epoch == 1:
            print(f'    epoch {epoch}, loss: {d2l.evaluate_loss(net, train_iter, loss):f}')

import torch.nn as nn

def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)

# Vanilla MLP architecture
def get_net(input_size):
    net = nn.Sequential(nn.Linear(input_size, 10),
                        nn.ReLU(),
                        nn.Linear(10, 1))
    net.apply(init_weights)
    return net

import torch.nn as nn
import torch


class MyRNN(nn.Module):

    def __init__(self, inputs, output):
        super(MyRNN, self).__init__()

        self.w1 = torch.rand(inputs, output)
        self.w2 = torch.rand(output, output)

        self.b = torch.zeros(1, output)

    def forward(self, x, x1):
        y = torch.mm(x, self.w1) + self.b
        y = torch.tanh(y)

        y1 = torch.mm(y, self.w2) + torch.mm(x1, y) + self.b

        return y, y1


N_INPUT = 4
N_NEURONS = 1

X0_batch = torch.tensor([[0, 1, 2, 0], [3, 4, 5, 0],
                         [6, 7, 8, 0], [9, 0, 1, 0]],
                        dtype=torch.float)  # t=0 => 4 X 4

X1_batch = torch.tensor([[9, 8, 7, 0], [0, 0, 0, 0],
                         [6, 5, 4, 0], [3, 2, 1, 0]],
                        dtype=torch.float)  # t=1 => 4 X 4

model = MyRNN(N_INPUT, N_NEURONS)

Y0_val, Y1_val = model(X0_batch, X1_batch)
print(Y0_val)
print(Y1_val)

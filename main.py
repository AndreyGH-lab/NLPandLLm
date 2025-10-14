#task 1
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# net_seq = nn.Sequential(
#     nn.Linear(in_features=3, out_features=5),
#     nn.Sigmoid(),
#     nn.Linear(in_features=5, out_features=2),
# )

# net_seq.to(device)
# x = torch.randn(1, 3).to(device)
# y_seq = net_seq(x)
# print("net_seq output:", y_seq)
# print()

# class Model(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features=3, out_features=5)
#         self.fc2 = nn.Linear(in_features=5, out_features=2)
#     def forward(self, x):
#         x = self.fc1(x)
#         return F.sigmoid(self.fc2(x))

# net_model = Model()
# net_model.to(device)
# y_model = net_model(x)
# print("net_model output:", y_model)
# print()

# total_params_seq = sum(p.numel() for p in net_seq.parameters() if p.requires_grad)
# total_params_model = sum(p.numel() for p in net_model.parameters() if p.requires_grad)

# print("trainable params")
# print(f"net_seq : {total_params_seq}")
# print(f"net_model : {total_params_model}") 

#programm output:
# net_seq output: tensor([[-0.2617,  0.6954]], grad_fn=<AddmmBackward0>)

# net_model output: tensor([[0.7213, 0.5068]], grad_fn=<SigmoidBackward0>)

# trainable params
# net_seq : 32
# net_model : 32

#---------------------------------------------------------------------------------------------

#task 2 

import numpy as np
import torch
class Neuron(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.func = torch.nn.Linear(1,1)
        self.func.weight.data = torch.tensor([[-10.0]])
        self.func.bias.data = torch.tensor([5.0])
    def forward(self, x):
        z = self.func(x)
        return torch.heaviside(z, torch.tensor([0.0]))
neuron = Neuron()
print(neuron.func.weight, neuron.func.bias)

x0 = torch.tensor([0.0])
x1 = torch.tensor([1.0])
print(neuron(x0))
print(neuron(x1))
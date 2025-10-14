import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

net_seq = nn.Sequential(
    nn.Linear(in_features=3, out_features=5),
    nn.Sigmoid(),
    nn.Linear(in_features=5, out_features=2),
)

net_seq.to(device)
x = torch.randn(1, 3).to(device)
y_seq = net_seq(x)
print("net_seq output:", y_seq)
print()

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(in_features=3, out_features=5)
        self.fc2 = nn.Linear(in_features=5, out_features=2)
    def forward(self, x):
        x = self.fc1(x)
        return F.sigmoid(self.fc2(x))

net_model = Model()
net_model.to(device)
y_model = net_model(x)
print("net_model output:", y_model)
print()

total_params_seq = sum(p.numel() for p in net_seq.parameters() if p.requires_grad)
total_params_model = sum(p.numel() for p in net_model.parameters() if p.requires_grad)

print("trainable params")
print(f"net_seq : {total_params_seq}")
print(f"net_model : {total_params_model}")

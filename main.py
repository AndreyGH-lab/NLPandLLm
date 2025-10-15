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

# import numpy as np
# import torch
# class Neuron(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.func = torch.nn.Linear(1,1)
#         self.func.weight.data = torch.tensor([[-10.0]])
#         self.func.bias.data = torch.tensor([5.0])
#     def forward(self, x):
#         z = self.func(x)
#         return torch.heaviside(z, torch.tensor([0.0]))
# neuron = Neuron()
# print(neuron.func.weight, neuron.func.bias)

# x0 = torch.tensor([0.0])
# x1 = torch.tensor([1.0])
# print(neuron(x0))
# print(neuron(x1))

#-------------------------------------------------

#task 3

# import torch

# class Neuron(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = torch.nn.Linear(2,1)
#         self.fc.weight.data = torch.tensor([[1.0, 1.0]])
#         self.fc.bias.data = torch.tensor([-1.5])
#     def forward(self, x):
#         z = self.fc(x)
#         return(torch.heaviside(z, torch.tensor([0.0])))
# neuron = Neuron()
# print(neuron.fc.weight, neuron.fc.bias)

# x00 = torch.tensor([0.0, 0.0])
# x01 = torch.tensor([0.0, 1.0])
# x10 = torch.tensor([1.0, 0.0])
# x11 = torch.tensor([1.0, 1.0])
# print(neuron(x00))
# print(neuron(x01))
# print(neuron(x10))
# print(neuron(x11))

#-----------------------------------------

#task4

# import torch

# class Neuron(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc = torch.nn.Linear(2,1)
#         self.fc.weight.data = torch.tensor([[1.0, 1.0]])
#         self.fc.bias.data = torch.tensor([-0.5])
#     def forward(self, x):
#         z = self.fc(x)
#         return(torch.heaviside(z, torch.tensor([0.0])))
# neuron = Neuron()
# print(neuron.fc.weight, neuron.fc.bias)

# x00 = torch.tensor([0.0, 0.0])
# x01 = torch.tensor([0.0, 1.0])
# x10 = torch.tensor([1.0, 0.0])
# x11 = torch.tensor([1.0, 1.0])
# print(neuron(x00))
# print(neuron(x01))
# print(neuron(x10))
# print(neuron(x11))

#-----------------------------------------------
#task 5

# import torch

# class XORNeuron(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.hidden = torch.nn.Linear(2,2)
#         self.output = torch.nn.Linear(2,1)

#         self.hidden.weight.data = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
#         self.hidden.bias.data = torch.tensor([-0.5, -1.5])

#         self.output.weight.data = torch.tensor([[1.0, -1.0]])
#         self.output.bias.data = torch.tensor([-0.5])
        
#     def forward(self, x):
#         hidden_out = torch.heaviside(self.hidden(x), torch.tensor([0.0]))
#         return torch.heaviside(self.output(hidden_out), torch.tensor([0.0]))
# neuron = XORNeuron()
# print(neuron.hidden.weight, neuron.hidden.bias)
# print(neuron.output.weight, neuron.output.bias)

# x00 = torch.tensor([0.0, 0.0])
# x01 = torch.tensor([0.0, 1.0])
# x10 = torch.tensor([1.0, 0.0])
# x11 = torch.tensor([1.0, 1.0])
# print(neuron(x00))
# print(neuron(x01))
# print(neuron(x10))
# print(neuron(x11))

#-------------------------------------

#task 6

# import torch

# class XOR(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#     def forward(self, x):
#         x1 = x[0]
#         x2 = x[1]

#         # первый нейрон — OR
#         n1 = torch.heaviside(1*x1 + 1*x2 - 0.5, torch.tensor([0.0]))

#         # второй нейрон — AND
#         n2 = torch.heaviside(1*x1 + 1*x2 - 1.5, torch.tensor([0.0]))

#         # третий нейрон — XOR = neuron( n1 - n2 - 0.5 )
#         y = torch.heaviside(1*n1 - 1*n2 - 0.5, torch.tensor([0.0]))
#         return y


# xor = XOR()

# x00 = torch.tensor([0.0, 0.0])
# x01 = torch.tensor([0.0, 1.0])
# x10 = torch.tensor([1.0, 0.0])
# x11 = torch.tensor([1.0, 1.0])

# print("(0, 0) :", xor(x00))
# print("(0, 1) :", xor(x01))
# print("(1, 0) :", xor(x10))
# print("(1, 1) :", xor(x11))

#-------------------------------------------------

#task 7

# import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"

# def find_max_tensor_size(dtype):
#     print(f"\n Проверяем тип {dtype}:")
#     n = 10_000  
#     step = n
#     last_ok = 0

#     while True:
#         try:
#             x = torch.empty((n,), dtype=dtype, device=device)
#             last_ok = n
#             n += step
#         except RuntimeError as e:
#             print(f"Ошибка при размере {n:,}: {e}")
#             break

#     print(f"Максимальный размер для {dtype}: {last_ok:,} элементов")
#     bytes_per_element = torch.tensor([], dtype=dtype).element_size()
#     total_mb = last_ok * bytes_per_element / 1024**2
#     print(f"~{total_mb:.2f} МБ памяти")
#     return last_ok

# results = {}
# for dtype in [torch.float32, torch.float64, torch.float16, torch.int32, torch.int64]:
#     results[str(dtype)] = find_max_tensor_size(dtype)

# print("\n Результаты:")
# for k, v in results.items():
#     print(f"{k}: {v:,} элементов")

#Ответы на вопросы:

# Вопрос 1: Только однослойные нейронные сети (перцептроны) могут иметь линейную разделяющую поверхность, 
# потому что у них нет скрытых слоёв и они могут проводить только прямую линию (или плоскость) между классами.

# Вопрос 2. Нет, не имеет смысла соединять такие нейроны с линейной функцией активации, 
# потому что даже если сделать несколько слоёв, всё равно получится одно линейное преобразование — сеть не 
# сможет обучиться сложным, нелинейным зависимостям.



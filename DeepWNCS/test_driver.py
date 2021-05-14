import torch

print(torch.cuda.is_available())
x = torch.rand(10)
print(x.size())

print("current cuda:", torch.cuda.current_device())
print("device name:", torch.cuda.get_device_name(0))
print("device count:", torch.cuda.device_count())
torch.cuda.device(1)
print("current cuda:", torch.cuda.current_device())
cuda1 = torch.device('cuda:1')

a = torch.tensor([1., 2.], device=cuda1)
print("a:",a)
print("test.....")
print("aaa")
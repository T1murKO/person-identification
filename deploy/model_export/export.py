import torch


model = torch.load('model.pth', torch.device('cpu'))
model.eval()
example_input = torch.rand(1, 3, 240, 240)
traced_script_module = torch.jit.trace(model, example_input)
traced_script_module.save("net.pt")
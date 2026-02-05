# compile_model.py
import torch
from model_initialize import CNN   # wherever your architecture lives

device = torch.device("cpu")

model = CNN()
model.load_state_dict(torch.load("website\\model\\model.pth", map_location=device))
model.eval()

scripted_model = torch.jit.script(model)
scripted_model.save("model.pt")

print("TorchScript model saved as model.pt")
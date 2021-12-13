import torch
from torchvision import transforms

transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
batch_size = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
learning_rate = 1e-2
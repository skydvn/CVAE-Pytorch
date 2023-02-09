from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

def get_MNIST(args):
    dataset = MNIST(
        root=args.data_dir, train=True, transform=transforms.ToTensor(),
        download=True)
    data_loader = DataLoader(
        dataset=dataset, batch_size=args.batch_size, shuffle=True)
    return data_loader
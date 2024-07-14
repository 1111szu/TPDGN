import torch


def get_device():
    device = torch.device('cpu' if not torch.cuda.is_available() else 'cuda')
    return device

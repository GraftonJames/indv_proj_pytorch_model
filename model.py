import os
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import datasets
from datasets import load_dataset, Video
from torchvision.transforms import ToTensor
from torchcodec.decoders import VideoDecoder

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3,3,5)
        self.conv2 = torch.nn.Conv2d(3,3,5)
        self.conv3 = torch.nn.Conv2d(3,3,5)
        self.conv4 = torch.nn.Conv2d(3,3,5)
        self.conv5 = torch.nn.Conv2d(3,3,5)

def get_host_info()
    thread_count = os.cpu_count()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    return (device, thread_count)

def get_dls(batch_size, device):
    ds = load_dataset("nexar-ai/nexar_collision_prediction")
    ds = ds.with_format("torch", device=device)

    train_ds = ds["train"]
    test_ds = ds["test"]

    #train_dl = DataLoader(train_ds, batch_size=batch_size)
    #test_dl = DataLoader(test_ds, batch_size=batch_size)


    return (train_ds, test_ds)


#### ENTRY POINT ####

(device, thread_count) = get_host_info()
(train_dl, test_dl) = get_dls(64, device)
print("####### LOADED DATASET #######")



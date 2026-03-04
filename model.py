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

class frame_nn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_b = torch.nn.Sequential(
            torch.nn.Conv2d(3,3,5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
    def forward(self, x):
        x = self.conv_b(x)
        x = self.conv_b(x)
        x = self.conv_b(x)
        x = self.conv_b(x)
        x = self.conv_b(x)

        return x

class temp_nn(nn.Module):
    def __init__(self, T, fcl_start):
        super().__init__()
        self.conv_1 = torch.nn.Conv1d(T, 1, T)
        self.conv_2 = torch.nn.Conv1d(T, 1, T)
        self.conv_3 = torch.nn.Conv1d(T, 1, T)

        self.fcl = torch.nn.Sequential(
            torch.nn.Linear(fcl_start, n2)
            torch.nn.ReLU()
            torch.nn.Linear(n2,n3)
            torch.nn.ReLU()
            torch.nn.Linear(n3,n4)
            torch.nn.ReLU()
            torch.nn.Linear(n4,n5)
            torch.nn.ReLU()
            torch.nn.Linear(n5,n6)
            torch.nn.ReLU()
        )

        self.softmax = torch.nn.Softmax(dim=0)
        
    def forward(self, x):
        #Seperate by channel
        x = torch.unbind(x, dim=2)
        #Apply 1d convs by channel
        x[0] = self.conv_1(x[0])
        x[1] = self.conv_2(x[1])
        x[2] = self.conv_3(x[2])

        #restack over channels
        x = torch.stack(x, dim=2)
        #remove time dimension (should be singular)
        x = torch.unbind(x, dim=1)[0]

        x = torch.flatten(x)
        x = self.softmax(x)

        return x

class main_nn(nn.Module):
    def __init__(self):
        super().__init__()
        frame_net = frame_nn()
        temp_net= temp_nn()

    def forward(self, x):
        B, T, C, H, W = x.shape

        #Seperates video into individual frames
        x = torch.unbind(x, dim=1)
        #Run through frame_nn
        x = self.frame_net(x)
        #Stack the feature result temorally
        x = torch.stack(x, dim=1)




        return x

def get_host_info():
    thread_count = os.cpu_count()
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    return (device, thread_count)

def get_dls(batch_size, device):
    ds = load_dataset("nexar-ai/nexar_collision_prediction")
    ds = ds.with_format("torch", device=device)

    train_ds = ds["train"]
    test_ds = ds["test"]

    train_dl = DataLoader(train_ds, batch_size=batch_size)
    test_dl = DataLoader(test_ds, batch_size=batch_size)


    return (train_dl, test_dl)


#### ENTRY POINT ####

(device, thread_count) = get_host_info()
(train_dl, test_dl) = get_dls(64, device)
print("####### LOADED DATASET #######")

print(train_dl.dataset["video"])

model = frame_nn().to(device)
print(model)


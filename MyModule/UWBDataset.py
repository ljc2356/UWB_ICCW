import torch
from torch.utils.data import Dataset

class UWBDataset(Dataset):
    def __init__(self,dataX,dataY):
        self.dataX = dataX
        self.dataY = dataY
        self.Num_dataset = dataX.shape[0]
    def __getitem__(self, item):
        oriCSI = torch.Tensor(self.dataX[item,:,:,:])
        labelCSI = torch.Tensor(self.dataY[item,:,:,:])
        return oriCSI,labelCSI
    def __len__(self):
        return self.Num_dataset
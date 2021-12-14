import itertools
import os.path
import torch
import h5py
import torch.nn
import numpy as np
from scipy.io import savemat
from scipy.io import loadmat
from torch.utils.data import DataLoader
from MyModule.UWBDataset import *
from MyModule.network_swinir import *
from MyModule.myLoss import *
from torch.utils.tensorboard import SummaryWriter

#%% init paremeters
dataset_folders = '../DataBase/20210820_ICCW/'
modelPath =  './resultModel/V2_1213/'
modelFileName = "EncDec.pth"

device = torch.device("cpu")
MSELossFun = torch.nn.MSELoss().to(device)

#%% read data from mat
testset_dict = h5py.File(name=os.path.join(dataset_folders,'testset.mat'),mode='r')
testset = np.array(testset_dict['test_dataset'])
testset = testset.swapaxes(0,3)
testset = testset.swapaxes(1,2)
    # init dataset from oridata
dataX_test = testset[:,0:2,:,0:50]
dataY_test = testset[:,2:4,:,0:50]
labelAngleFromData = testset[:,2,1,50].reshape(-1,1)   #data is clean
    # init dataloader
dataloader_test = DataLoader(
    dataset=UWBDataset(dataX_test,dataY_test),
    batch_size= 100,
    shuffle = False,
    num_workers=1
)

#%% init Model
EncDec = UWBSwinIR().to(device)

    #load from history
EncDec.load_state_dict(torch.load(os.path.join(modelPath,modelFileName),map_location=torch.device('cpu')))

#read data from test Dataloader
with torch.no_grad():
    testBatchIndex = 0
    lossSumTestBatchs = 0
    # init result
    oriAngleFromMUSIC = np.empty([0, 1])
    labelAngleFromMUSIC = np.empty([0, 1])
    reconAngleFromMUSIC = np.empty([0, 1])
    # dataloader_test_iter = dataloader_test.__iter__()
    # oriCSITestBatch, labelCSITestBatch = next(dataloader_test_iter)

    for oriCSITestBatch,labelCSITestBatch in dataloader_test:

        if torch.cuda.is_available():
            oriCSITestBatch = oriCSITestBatch.cuda()
            labelCSITestBatch = labelCSITestBatch.cuda()
        #testing
        reconCSITestBatch = EncDec(oriCSITestBatch)
        ## test Angle
        oriAngleFromMUSIC = np.vstack((oriAngleFromMUSIC,(MUSIC(oriCSITestBatch)).reshape(-1,1)))
        labelAngleFromMUSIC = np.vstack((labelAngleFromMUSIC,(MUSIC(labelCSITestBatch)).reshape(-1,1)))
        reconAngleFromMUSIC = np.vstack((reconAngleFromMUSIC,(MUSIC(reconCSITestBatch)).reshape(-1,1)))
        #print iter
        testBatchIndex = testBatchIndex + 1
        print(testBatchIndex)
    meanLossOri = np.mean(np.abs(wrapToPi(oriAngleFromMUSIC - labelAngleFromData)))
    meanLossRecon = np.mean(np.abs(wrapToPi(reconAngleFromMUSIC - labelAngleFromData)))
    meanLossLabel = np.mean(np.abs(wrapToPi(labelAngleFromMUSIC- labelAngleFromData)))
    print("meanLossOri : {}".format(meanLossOri))
    print("meanLossRecon : {}".format(meanLossRecon))
    print("meanLossLabel : {}".format(meanLossLabel))









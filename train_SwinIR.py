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
writer = SummaryWriter(log_dir="./logs/V2_1213")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LossFun = torch.nn.L1Loss().to(device)
num_epochs = 2000
loss_test_mat = np.zeros(shape=(num_epochs,1))

#%% read data from mat
trainset_dict = h5py.File(name=os.path.join(dataset_folders,'trainset.mat'),mode='r')
trainset = np.array(trainset_dict['train_dataset'])
trainset = trainset.swapaxes(0,3)
trainset = trainset.swapaxes(1,2)

testset_dict = h5py.File(name=os.path.join(dataset_folders,'testset.mat'),mode='r')
testset = np.array(testset_dict['test_dataset'])
testset = testset.swapaxes(0,3)
testset = testset.swapaxes(1,2)
    # init dataset from oridata
dataX_train = trainset[:,0:2,:,0:50]
dataY_train = trainset[:,2:4,:,0:50]

dataX_test = testset[:,0:2,:,0:50]
dataY_test = testset[:,2:4,:,0:50]
    # init dataloader
dataloader_train = DataLoader(
    dataset=UWBDataset(dataX_train,dataY_train),
    batch_size = 100,
    shuffle = True,
    num_workers= 500
)

dataloader_test = DataLoader(
    dataset=UWBDataset(dataX_test,dataY_test),
    batch_size= 100,
    shuffle = False,
    num_workers=1
)

#%% init Model and optimizers
EncDec = UWBSwinIR().to(device)

#     # load from history
# Enc.load_state_dict(torch.load(os.path.join(modelPath,EncFileName)))
# Dec.load_state_dict(torch.load(os.path.join(modelPath,DecFileName)))

optimizer = torch.optim.Adam(
    params = EncDec.parameters(),
    lr=0.001
)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.99)

#%% train
for epoch in range(num_epochs):
    #read batch from Training Dataloader
    batchIndex = 0
    lossSumBatchs = 0
    #%%
    # oriCSIBatch, labelCSIBatch = next(dataloader_train.__iter__())
    for oriCSIBatch,labelCSIBatch in dataloader_train:
        if torch.cuda.is_available():
            oriCSIBatch = oriCSIBatch.cuda()
            labelCSIBatch = labelCSIBatch.cuda()

        #start training
        optimizer.zero_grad()
        reconCSIBatch = EncDec(oriCSIBatch)

        #compute loss
        loss2Self = amplitudeLoss(reconCSIBatch,oriCSIBatch,LossFun) \
                       + pdoaLoss(reconCSIBatch,oriCSIBatch,LossFun)
        loss2Label = amplitudeLoss(reconCSIBatch,labelCSIBatch,LossFun) \
                     + pdoaLoss(reconCSIBatch,labelCSIBatch,LossFun)
        # loss = loss2Self + loss2Label
        loss = loss2Label
        lossSumBatchs = lossSumBatchs + loss.item()
        print("Training epochs: {}, batchs: {}, loss2Self: {}, loss2Label: {}, loss: {}".format(
            epoch,batchIndex,loss2Self,loss2Label,loss
        ))
        batchIndex += 1

        #gradient descent
        loss.backward()
        optimizer.step()
    #%%
    lossSumBatchs = lossSumBatchs/(batchIndex+1)  #avg training loss
    writer.add_scalar(tag='lossTrain',scalar_value=lossSumBatchs,global_step=epoch)

    #decent learning rate every 10 epochs
    if (epoch%10) == 0:
        scheduler.step()

    #read data from test Dataloader
    with torch.no_grad():
        testBatchIndex = 0
        lossSumTestBatchs = 0
        for oriCSITestBatch,labelCSITestBatch in dataloader_test:
            if torch.cuda.is_available():
                oriCSITestBatch = oriCSITestBatch.cuda()
                labelCSITestBatch = labelCSITestBatch.cuda()

            #testing
            reconCSITestBatch = EncDec(oriCSITestBatch)
            #compute test loss
            loss2SelfTest = amplitudeLoss(reconCSITestBatch,oriCSITestBatch,LossFun) \
                            + pdoaLoss(reconCSITestBatch,oriCSITestBatch,LossFun)
            loss2LabelTest = amplitudeLoss(reconCSITestBatch,labelCSITestBatch,LossFun) \
                             + pdoaLoss(reconCSITestBatch,labelCSITestBatch,LossFun)
            # lossTest = loss2SelfTest + loss2LabelTest
            lossTest = loss2LabelTest
            lossSumTestBatchs = lossSumTestBatchs + lossTest.item()
            print("Testing epochs: {}, batchs: {}, loss2Self: {}, loss2Label: {}, loss: {}".format(
                epoch, testBatchIndex, loss2SelfTest, loss2LabelTest, lossTest
            ))
            testBatchIndex = testBatchIndex + 1

    lossSumTestBatchs = lossSumTestBatchs / (testBatchIndex + 1) #avg test loss
    writer.add_scalar(tag="lossTest",scalar_value=lossSumTestBatchs,global_step=epoch)

    #save model
    loss_test_mat[epoch,0] = lossSumTestBatchs
    loss_test_mat[epoch,0] = lossSumTestBatchs
    # if (epoch > 1) & (loss_test_mat[epoch,0]<loss_test_mat[epoch-1,0]):
    torch.save(EncDec.state_dict(),os.path.join(modelPath,modelFileName))











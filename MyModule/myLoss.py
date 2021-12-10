import torch
import numpy as np
from MyModule.myTools import *

def amplitudeLoss(X,Y,lossFun = torch.nn.MSELoss().to(torch.device("cpu"))):
    xAmp = torch.sqrt(torch.square(input=X[:, 0, :, :]) + torch.square(input=X[:, 1, :, :]))
    yAmp = torch.sqrt(torch.square(input=Y[:, 0, :, :]) + torch.square(input=Y[:, 1, :, :]))
    return lossFun(xAmp,yAmp)

def pdoaLoss(X,Y,lossFun = torch.nn.MSELoss().to(torch.device("cpu"))):
    xPhase = torch.atan2(input=X[:,1,:,:],other=(X[:,0,:,:]))
    xPDOA = xPhase - xPhase[:,0,:].view(-1,1,50)

    yPhase = torch.atan2(input=Y[:,1,:,:],other=(Y[:,0,:,:]))
    yPDOA = yPhase - yPhase[:,0,:].view(-1,1,50)    #here should wrapToPi and lossfunc should replace
    xyDiff = wrapToPi(xPDOA[:,:,0:10] - yPDOA[:,:,0:10])  #focus on first ten
    zeroMat = torch.zeros(size=xyDiff.shape).to(xyDiff.device)
    return lossFun(xyDiff,zeroMat)



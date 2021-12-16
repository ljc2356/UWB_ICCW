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
    xyDiff = wrapToPi(xPDOA[:,:,0:20] - yPDOA[:,:,0:20])  #focus on first ten
    zeroMat = torch.zeros(size=xyDiff.shape).to(xyDiff.device)
    return lossFun(xyDiff,zeroMat)

def pdoaItemLoss(X,PDOA,Weight = None,lossFun = torch.nn.MSELoss().to(torch.device("cpu"))):
    xPhase = torch.atan2(input=X[:,1,:,:],other=X[:,0,:,:])
    xPDOA = xPhase - xPhase[:,0,:].view(-1,1,50)
    esp = 0.00001
    if Weight == None:
        xyDiff = wrapToPi(xPDOA[:,:,0:20] - PDOA.view(PDOA.shape[0],8,1))
        zeroMat = torch.zeros(size=xyDiff.shape).to(xyDiff.device)
    else:
        xyDiff = wrapToPi(xPDOA[:, :, 0:50] - PDOA.view(PDOA.shape[0], 8, 1))   #(N,8,50)
        # maxWeight = torch.max(Weight,dim=1).values.view(-1,1)    #(N,50)->(N)
        #
        # rvsMaxWeight = torch.pow(input=maxWeight+esp,exponent=-1) #(N)
        # Weight = torch.mul(input=Weight,other=rvsMaxWeight)   #(N,50)
        Weight = 50 * Weight.view(Weight.shape[0],1,Weight.shape[1])   #(N,50) -> (N,1,50)
        xyDiff = torch.mul(input=xyDiff,other=Weight)
        zeroMat = torch.zeros(size=xyDiff.shape).to(xyDiff.device)
    return lossFun(xyDiff,zeroMat)





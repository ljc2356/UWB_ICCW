import numpy as np
import math

def wrapToPi(angle):
    result = (angle + math.pi) % (math.tau) - math.pi
    return result

def sterrArray(theta,fc = 3.9936e+9,c = 299792458,radius = 0.0732):
    pi = math.pi
    numAngles = theta.shape[0]
    A = np.array(np.zeros((8,numAngles)),dtype='complex64')
    alpha = np.zeros(8)
    for i in range(8):
        alpha[i] = wrapToPi((i)*pi/4)
        A[i,0:numAngles] = np.exp(1j * (2 * pi * fc / c * radius * np.cos(theta - alpha[i]))).reshape(1,numAngles)
    return A

def AOAcal(Pdoa,fs = 0.001,fc = 3.9936e+9,c = 299792458,radius = 0.0732,low_threshold = -1*math.pi,high_threshold = math.pi):
    Pdoa = Pdoa.numpy()
    pi = math.pi
    alpha = []
    for i in range(8):
        alpha.append(wrapToPi(i * pi / 4))

    theta = np.arange(low_threshold,high_threshold,fs)
    std_phi = 2 * pi * fc / c * radius * np.cos(theta)

    phi = np.zeros((8,len(theta)))
    predict_pdoa = np.zeros((8,len(theta)))
    for i in range(8):
        phi[i,:] = 2 * pi *fc /c *radius * np.cos(theta - alpha[i])
        predict_pdoa[i,:] = wrapToPi(phi[i,:] - std_phi.reshape([1,-1])).reshape(1,-1)

    pdoa_diff = wrapToPi(predict_pdoa.transpose() - Pdoa.transpose())
    pdoa_square = pdoa_diff * pdoa_diff
    pdoa_sum_square = pdoa_square.sum(axis=1)
    target_index = np.argmin(pdoa_sum_square)
    theta_est = theta[target_index]

    return theta_est


def MUSIC(CSI,Weight = None,fc = 3.9936e+9,c = 299792458,radius = 0.0732,low_threshold = -1*math.pi,high_threshold = math.pi):
    pi = math.pi
    realCSI = CSI[:,0,:,:].cpu().detach().numpy()
    imagCSI = CSI[:,1,:,:].cpu().detach().numpy()
    npCSI = np.array(realCSI + 1j * imagCSI)   #clear
    R = np.array(np.zeros(shape=(npCSI.shape[0],8,8)),dtype='complex64')
    estAngle = np.zeros((npCSI.shape[0]))
    maxPath = 20

    if Weight == None:  #if Weight equals None,no attention
        flagIndex = np.zeros(shape=(npCSI.shape[0],50))
        flagIndex[:,0:maxPath] = 1
    else:
        Weight = Weight.cpu().detach().numpy()
        for dataIndex in range(npCSI.shape[0]):
            Weight[dataIndex,:] = Weight[dataIndex,:] / np.max(Weight[dataIndex,:])
        flagIndex = np.zeros(Weight.shape)
        flagIndex[np.where(Weight>0.7)] += 1    #recongnize los path

    for dataIndex in range(npCSI.shape[0]):
        for freIndex in range(maxPath):
            R[dataIndex,0:8,0:8] = R[dataIndex,0:8,0:8] \
                                   + flagIndex[dataIndex,freIndex] * np.matmul(
                npCSI[dataIndex,:,freIndex].reshape(8,1),
                npCSI[dataIndex,:,freIndex].reshape(8,1).conj().T
            )
        U,S,Vh = np.linalg.svd(R[dataIndex,:,:])
        En = U[:,1:]
        theta = np.array(np.arange(start=-1 * pi,stop = pi,step = 0.01))
        A = sterrArray(theta)
        Pmu = 1./np.diag(A.conj().T @ En @ En.conj().T @ A)
        Pmu = np.abs(Pmu)
        estAngle[dataIndex] = theta[np.argmax(Pmu)]
    return estAngle







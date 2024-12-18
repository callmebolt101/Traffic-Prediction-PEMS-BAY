from operator import ge
import sys
import os
import shutil
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import torch
import torch.nn as nn
import Metrics
from GMAN import *
from Param import *
from Param_GMAN import *
import Utils


def getXSYS(data, mode):
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS = [], []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            XS.append(x), YS.append(y)
    XS, YS = np.array(XS), np.array(YS)
    return XS, YS


def getTE(df, mode):
    time = pd.DatetimeIndex(df.index)
    dayofweek = np.reshape(np.array(time.weekday), (-1, 1))
    timeofday = (df.index.values - df.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    timeofday = np.reshape(timeofday, (-1, 1))
    time = np.concatenate((dayofweek, timeofday), -1)
    TRAIN_NUM = int(time.shape[0] * TRAINRATIO)
    TE = []
    if mode == 'TRAIN':    
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            t = time[i:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            TE.append(t)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  time.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            t = time[i:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            TE.append(t)
    TE = np.array(TE)
    return TE


def split_peak_offpeak(data, df):
    time = pd.DatetimeIndex(df.index)
    hour = time.hour
    peak_indices = ((hour >= 7) & (hour < 9)) | ((hour >= 17) & (hour < 19))
    off_peak_indices = ~peak_indices

    peak_data = data[peak_indices]
    off_peak_data = data[off_peak_indices]

    return peak_data, off_peak_data


def getModel(name, device):
    SE = getSE(SEPATH).to(device=device)
    model = GMAN(SE, TIMESTEP_IN, device).to(device)
    return model


def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, te, y in data_iter:
            y_pred = model(x, te)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n


def trainModel(name, mode, XS, TE, YS, device):
    print('Model Training Started ...', time.ctime())
    model = getModel(name, device)
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    TE_torch = torch.Tensor(TE).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)

    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    for epoch in range(EPOCH):
        loss_sum, n = 0.0, 0
        model.train()
        for x, te, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x, te)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        print(f"Epoch {epoch}, Train Loss: {train_loss}, Val Loss: {val_loss}")


def testModel(name, mode, XS, TE, YS, device):
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    TE_torch = torch.Tensor(TE).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, TE_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name, device)
    model.load_state_dict(torch.load(PATH+ '/' + name + '.pt'))
    torch_score = evaluateModel(model, criterion, test_iter)
    print(f"Test Torch MSE: {torch_score}")


def train_and_test_peak_offpeak(model_name, data, df, device):
    """
    Train and test the model on peak and off-peak data.
    """
    print("\n--- Splitting data into peak and off-peak periods ---")
    peak_data, off_peak_data = split_peak_offpeak(data, df)
    print(f"Peak data shape: {peak_data.shape}, Off-peak data shape: {off_peak_data.shape}")

    for mode, dataset in zip(['peak', 'off-peak'], [peak_data, off_peak_data]):
        print(f"\n--- Training and Testing for {mode.upper()} Data ---")

        XS, YS = getXSYS(dataset, 'TRAIN')
        TE = getTE(df.loc[dataset.index], "TRAIN")
        trainModel(f"{model_name}_{mode}", "train", XS, TE, YS, device)

        XS_test, YS_test = getXSYS(dataset, 'TEST')
        TE_test = getTE(df.loc[dataset.index], "TEST")
        testModel(f"{model_name}_{mode}", "test", XS_test, TE_test, YS_test, device)


################# Main Code #######################
MODELNAME = 'GMAN'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
torch.manual_seed(100)
np.random.seed(100)
df = pd.read_hdf(FLOWPATH)
data = df.values
scaler = StandardScaler()
data = scaler.fit_transform(data)


def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    print("\n--- Main Training and Testing ---")
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    trainTE = getTE(df, "TRAIN")
    trainModel(MODELNAME, "train", trainXS, trainTE, trainYS, device)

    testXS, testYS = getXSYS(data, 'TEST')
    testTE = getTE(df, "TEST")
    testModel(MODELNAME, "test", testXS, testTE, testYS, device)

    print("\n--- Running Peak and Off-Peak Model Training ---")
    train_and_test_peak_offpeak(MODELNAME, data, df, device)


if __name__ == '__main__':
    main()

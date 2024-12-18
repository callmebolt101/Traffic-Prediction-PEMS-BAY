import sys
import os
import shutil
import math
import numpy as np
import pandas as pd
import scipy.sparse as ss
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import time
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
import Metrics
import Utils
from GraphWaveNet import *
from Param import *
from Param_GraphWaveNet import *

def getTimestamp(data):
    # data is a pandas dataframe with timestamp ID.
    data_feature = data.values.reshape(data.shape[0],data.shape[1],1)
    feature_list = [data_feature]
    num_samples, num_nodes = data.shape
    time_ind = (data.index.values - data.index.values.astype("datetime64[D]")) / np.timedelta64(1, "D")
    time_in_day = np.tile(time_ind, [num_nodes,1]).transpose((1, 0))
    return time_in_day

def getXSYSTIME(data, data_time, mode):
    # When CHANNENL = 2, use this function to get data plus time as two channels.
    # data: numpy, data_time: numpy from getTimestamp
    TRAIN_NUM = int(data.shape[0] * TRAINRATIO)
    XS, YS, XS_TIME = [], [], []
    if mode == 'TRAIN':
        for i in range(TRAIN_NUM - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            t = data_time[i:i+TIMESTEP_IN, :]
            XS.append(x), YS.append(y), XS_TIME.append(t)
    elif mode == 'TEST':
        for i in range(TRAIN_NUM - TIMESTEP_IN,  data.shape[0] - TIMESTEP_OUT - TIMESTEP_IN + 1):
            x = data[i:i+TIMESTEP_IN, :]
            y = data[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
            t = data_time[i:i+TIMESTEP_IN, :]
            XS.append(x), YS.append(y), XS_TIME.append(t)
    XS, YS, XS_TIME = np.array(XS), np.array(YS), np.array(XS_TIME)
    XS = np.concatenate([np.expand_dims(XS, axis=-1), np.expand_dims(XS_TIME, axis=-1)], axis=-1)
    XS, YS = XS.transpose(0, 3, 2, 1), YS[:, :, :, np.newaxis]
    return XS, YS

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
    XS, YS = XS[:, :, :, np.newaxis], YS[:, :, :, np.newaxis]
    XS = XS.transpose(0, 3, 2, 1)
    return XS, YS

def getModel(name):
    # way1: only adaptive graph.
    # model = gwnet(device, num_nodes = N_NODE, in_dim=CHANNEL).to(device)
    # return model
    
    # way2: adjacent graph + adaptive graph
    adj_mx = load_adj(ADJPATH, ADJTYPE)
    supports = [torch.tensor(i).to(device) for i in adj_mx]
    model = gwnet(device, num_nodes=N_NODE, in_dim=CHANNEL, supports=supports).to(device)
    return model

def evaluateModel(model, criterion, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x)
            l = criterion(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

def predictModel(model, data_iter):
    YS_pred = []
    model.eval()
    with torch.no_grad():
        for x, y in data_iter:
            YS_pred_batch = model(x)
            YS_pred_batch = YS_pred_batch.cpu().numpy()
            YS_pred.append(YS_pred_batch)
        YS_pred = np.vstack(YS_pred)
    return YS_pred

def trainModel(name, mode, XS, YS):
    print('Model Training Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    model = getModel(name)
    # summary(model, (CHANNEL,N_NODE,TIMESTEP_IN), device=device)
    summary(model, (CHANNEL, N_NODE, TIMESTEP_IN), device="cuda" if torch.cuda.is_available() else "cpu")
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    trainval_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    trainval_size = len(trainval_data)
    train_size = int(trainval_size * (1-TRAINVALSPLIT))
    print('XS_torch.shape:  ', XS_torch.shape)
    print('YS_torch.shape:  ', YS_torch.shape)
    train_data = torch.utils.data.Subset(trainval_data, list(range(0, train_size)))
    val_data = torch.utils.data.Subset(trainval_data, list(range(train_size, trainval_size)))
    train_iter = torch.utils.data.DataLoader(train_data, BATCHSIZE, shuffle=True)
    val_iter = torch.utils.data.DataLoader(val_data, BATCHSIZE, shuffle=True)
    
    min_val_loss = np.inf
    wait = 0

    print('LOSS is :',LOSS)
    if LOSS == "MaskMAE":
        criterion = Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    if OPTIMIZER == 'RMSprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARN)
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARN)
    
    for epoch in range(EPOCH):
        starttime = datetime.now()
        loss_sum, n = 0.0, 0
        model.train()
        for x, y in train_iter:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * y.shape[0]
            n += y.shape[0]
        train_loss = loss_sum / n
        val_loss = evaluateModel(model, criterion, val_iter)
        if val_loss < min_val_loss:
            wait = 0
            min_val_loss = val_loss
            torch.save(model.state_dict(), PATH + '/' + name + '.pt')
        else:
            wait += 1
            if wait == PATIENCE:
                print('Early stopping at epoch: %d' % epoch)
                break
        endtime = datetime.now()
        epoch_time = (endtime - starttime).seconds
        print("epoch", epoch, "time used:", epoch_time," seconds ", "train loss:", train_loss, "validation loss:", val_loss)
        with open(PATH + '/' + name + '_log.txt', 'a') as f:
            f.write("%s, %d, %s, %d, %s, %s, %.10f, %s, %.10f\n" % ("epoch", epoch, "time used", epoch_time, "seconds", "train loss", train_loss, "validation loss:", val_loss))
            
    # Predicting on training data
    torch_score = evaluateModel(model, nn.L1Loss(), train_iter)
    YS_pred = predictModel(model, torch.utils.data.DataLoader(trainval_data, BATCHSIZE, shuffle=False))
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)

    # Flattening and reshaping data for inverse transformation
    num_samples, num_timesteps, num_nodes, _ = YS.shape
    YS_flat = YS.reshape(-1, num_nodes)
    YS_pred_flat = YS_pred.reshape(-1, num_nodes)

    # Performing inverse transformation
    YS_inv = scaler.inverse_transform(YS_flat)
    YS_pred_inv = scaler.inverse_transform(YS_pred_flat)

    # Reshaping back to original dimensions
    YS = YS_inv.reshape(num_samples, num_timesteps, num_nodes, 1)
    YS_pred = YS_pred_inv.reshape(num_samples, num_timesteps, num_nodes, 1)

    # Calculating metrics
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write(f"{name}, {mode}, Torch MSE, {torch_score:.10e}, {torch_score:.10f}\n")
        f.write(f"{name}, {mode}, MSE, RMSE, MAE, MAPE, {MSE:.10f}, {RMSE:.10f}, {MAE:.10f}, {MAPE:.10f}\n")
    
    print('*' * 40)
    print(f"{name}, {mode}, Torch MSE, {torch_score:.10e}, {torch_score:.10f}")
    print(f"{name}, {mode}, MSE, RMSE, MAE, MAPE, {MSE:.10f}, {RMSE:.10f}, {MAE:.10f}, {MAPE:.10f}")
    print('Model Training Ended ...', time.ctime())
        
def testModel(name, mode, XS, YS):
    if LOSS == "MaskMAE":
        criterion = Utils.masked_mae
    if LOSS == 'MSE':
        criterion = nn.MSELoss()
    if LOSS == 'MAE':
        criterion = nn.L1Loss()
    print('Model Testing Started ...', time.ctime())
    print('TIMESTEP_IN, TIMESTEP_OUT', TIMESTEP_IN, TIMESTEP_OUT)
    
    XS_torch, YS_torch = torch.Tensor(XS).to(device), torch.Tensor(YS).to(device)
    test_data = torch.utils.data.TensorDataset(XS_torch, YS_torch)
    test_iter = torch.utils.data.DataLoader(test_data, BATCHSIZE, shuffle=False)
    model = getModel(name)
    model.load_state_dict(torch.load('../save/pred_PEMS-BAY_GraphWaveNet_2412160654/GraphWaveNet.pt'))

    torch_score = evaluateModel(model, criterion, test_iter)
    YS_pred = predictModel(model, test_iter)
    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)

    # Flatten for scaler.inverse_transform()
    num_samples, num_timesteps, num_nodes, _ = YS.shape
    YS_flat = YS.reshape(-1, num_nodes)  # Shape: (samples * timesteps, nodes)
    YS_pred_flat = YS_pred.reshape(-1, num_nodes)  # Same shape

    # Inverse transformation
    YS_inv = scaler.inverse_transform(YS_flat)  # Shape: (samples * timesteps, nodes)
    YS_pred_inv = scaler.inverse_transform(YS_pred_flat)

    # Reshape back to original 3D shape
    YS = YS_inv.reshape(num_samples, num_timesteps, num_nodes, 1)
    YS_pred = YS_pred_inv.reshape(num_samples, num_timesteps, num_nodes, 1)

    print('YS.shape, YS_pred.shape,', YS.shape, YS_pred.shape)
    np.save(PATH + '/' + MODELNAME + '_prediction.npy', YS_pred)
    np.save(PATH + '/' + MODELNAME + '_groundtruth.npy', YS)

    # Evaluate performance
    MSE, RMSE, MAE, MAPE = Metrics.evaluate(YS, YS_pred)
    print('*' * 40)
    print(f"{name}, {mode}, Torch MSE, {torch_score:.10e}, {torch_score:.10f}")
    
    with open(PATH + '/' + name + '_prediction_scores.txt', 'a') as f:
        f.write(f"{name}, {mode}, Torch MSE, {torch_score:.10e}, {torch_score:.10f}\n")
        f.write(f"{name}, {mode}, MSE, RMSE, MAE, MAPE, {MSE:.10f}, {RMSE:.10f}, {MAE:.10f}, {MAPE:.10f}\n")

    print(f"all pred steps, {name}, {mode}, MSE, RMSE, MAE, MAPE, {MSE:.10f}, {RMSE:.10f}, {MAE:.10f}, {MAPE:.10f}")
    print('Model Testing Ended ...', time.ctime())
        
################# Parameter Setting #######################
MODELNAME = 'GraphWaveNet'
KEYWORD = 'pred_' + DATANAME + '_' + MODELNAME + '_' + datetime.now().strftime("%y%m%d%H%M")
PATH = '../save/' + KEYWORD
torch.manual_seed(100)
torch.cuda.manual_seed(100)
np.random.seed(100)
# torch.backends.cudnn.deterministic = True
###########################################################
# GPU = sys.argv[-1] if len(sys.argv) == 2 else '3'
GPU = 0
device = torch.device("cuda:{}".format(GPU)) if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}")
print(torch.cuda.is_available())  # Should print True
print(torch.cuda.device_count())  # Should print 1 if one GPU is available
###########################################################
data = pd.read_hdf(FLOWPATH).values
scaler = StandardScaler()
data = scaler.fit_transform(data)
print('data.shape', data.shape)
###########################################################

# Function to run model on peak and off-peak data
def run_model_for_peak_offpeak(file_path):
    """
    Splits the dataset into peak and off-peak, trains models, evaluates, and saves weights.
    """
    # Step 1: Split dataset
    peak_data, off_peak_data = split_peak_offpeak(file_path)
    print("Dataset split into peak and off-peak successfully.")
    
    # Step 2: Initialize model
    model_peak = GraphWaveNet()
    model_off_peak = GraphWaveNet()
    
    # Step 3: Train and evaluate for peak data
    print("\n--- Training on PEAK data ---")
    train_model(model_peak, peak_data)
    metrics_peak = evaluatemodel(model_peak, peak_data)
    
    # Step 4: Train and evaluate for off-peak data
    print("\n--- Training on OFF-PEAK data ---")
    train_model(model_off_peak, off_peak_data)
    metrics_off_peak = evaluatemodel(model_off_peak, off_peak_data)
    
    # Step 5: Save model weights
    torch.save(model_peak.state_dict(), "peak_model_weights.pth")
    torch.save(model_off_peak.state_dict(), "off_peak_model_weights.pth")
    print("\nModel weights saved: 'peak_model_weights.pth' and 'off_peak_model_weights.pth'")
    
    # Print final results
    print("\n--- Final Metrics ---")
    print("PEAK Data Metrics:", metrics_peak)
    print("OFF-PEAK Data Metrics:", metrics_off_peak)
    
    
def main():
    if not os.path.exists(PATH):
        os.makedirs(PATH)
    currentPython = sys.argv[0]
    shutil.copy2(currentPython, PATH)
    shutil.copy2('GraphWaveNet.py', PATH)
    shutil.copy2('Param.py', PATH)
    shutil.copy2('Param_GraphWaveNet.py', PATH)
    
    print(KEYWORD, 'training started', time.ctime())
    trainXS, trainYS = getXSYS(data, 'TRAIN')
    print('TRAIN XS.shape YS.shape:', trainXS.shape, trainYS.shape)
    trainModel(MODELNAME, 'train', trainXS, trainYS)
    
    print(KEYWORD, 'testing started', time.ctime())
    testXS, testYS = getXSYS(data, 'TEST')
    print('TEST XS.shape YS.shape:', testXS.shape, testYS.shape)
    testModel(MODELNAME, 'test', testXS, testYS)
    
    # New functionality: Run model for peak and off-peak data
    print("\n--- Running Peak and Off-Peak Model Training ---")
    run_model_for_peak_offpeak(PATH)
    print("--- Peak and Off-Peak Model Training Complete ---")
    
    


    
if __name__ == '__main__':
    main()

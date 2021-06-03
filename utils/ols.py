# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import random
import argparse
import itertools
import nibabel as nib
from nilearn import plotting
from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import torch
import time
import pickle



def vectorized_correlation(x,y):
    # Pearson Correlation
    dim = 0

    centered_x = x - x.mean(axis=dim, keepdims=True)
    centered_y = y - y.mean(axis=dim, keepdims=True)

    covariance = (centered_x * centered_y).sum(axis=dim, keepdims=True)

    bessel_corrected_covariance = covariance / (x.shape[dim] - 1)

    x_std = x.std(axis=dim, keepdims=True)+1e-8
    y_std = y.std(axis=dim, keepdims=True)+1e-8

    corr = bessel_corrected_covariance / (x_std * y_std)
    return corr.ravel()


class OLS_pytorch(object):
    def __init__(self,use_gpu=False):
        self.coefficients = []  # 相当于θ，
        self.use_gpu = use_gpu
        self.X = None
        self.y = None

    def fit(self, X, y):  # 训练
        # X -> train_activations, y -> fmri_train
        if len(X.shape) == 1:  # 加一个维度
            X = self._reshape_x(X) # flatten -> [num_element, 1]
        if len(y.shape) == 1:
            y = self._reshape_x(y) # flatten -> [num_element, 1]

        X =  self._concatenate_ones(X)  # [1, x_1, x_2, ..., x_n]

        X = torch.from_numpy(X).float()
        y = torch.from_numpy(y).float()
        if self.use_gpu:
            X = X.cuda()
            y = y.cuda()
        # matmul -> 带broadcast的矩阵乘法
        XtX = torch.matmul(X.t(), X)  # [101,900]*[900,101] -> [101,101]
        Xty = torch.matmul(X.t(), y.unsqueeze(2))  # [101,900]*[368,900,1] -> [368,101,1]
        XtX = XtX.unsqueeze(0) # [1, 101, 101]
        # repeat_interleave 复制函数，按维度复制规定次数
        XtX = torch.repeat_interleave(XtX, y.shape[0], dim=0) # result -> [368, 101, 101]
        # torch.solve(B,A) -> return the solution of AX=B 
        betas_cholesky, _ = torch.solve(Xty, XtX)  # betas [368,101,1] _[368,101,101]
        
        self.coefficients = betas_cholesky

    def predict(self, entry):
        if len(entry.shape) == 1:
            entry = self._reshape_x(entry)
        entry =  self._concatenate_ones(entry)
        entry = torch.from_numpy(entry).float()
        if self.use_gpu:
            entry = entry.cuda()
        prediction = torch.matmul(entry, self.coefficients) # [100, 101]*[368,101,1] -> [368, 100, 1]
        prediction = prediction.cpu().numpy()
        prediction = np.squeeze(prediction).T # -> [368, 100] -> [100, 368]
        return prediction

    def score(self):
        prediction = torch.matmul(self.X, self.coefficients)
        prediction = prediction
        yhat = prediction
        ybar = (torch.sum(self.y,dim=1, keepdim=True)/self.y.shape[1]).unsqueeze(2)
        ssreg = torch.sum((yhat-ybar)**2,dim=1, keepdim=True)
        sstot = torch.sum((self.y.unsqueeze(2) - ybar)**2,dim=1, keepdim=True)
        score = ssreg / sstot
        return score.cpu().numpy().ravel()

    def _reshape_x(self,X):
        return X.reshape(-1,1)  # 拉成(num_element, 1)的形状

    def _concatenate_ones(self,X):
        ones = np.ones(shape=X.shape[0]).reshape(-1,1)
        return np.concatenate((ones,X),1)


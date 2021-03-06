# -*- coding: utf-8 -*-
import numpy as np
import os
import glob
import time
import random
import argparse
import itertools
import nibabel as nib
from nilearn import plotting
from numpy.matrixlib.defmatrix import _from_string
from tqdm import tqdm
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
import time
import pickle
from tqdm import tqdm
from utils.ols import vectorized_correlation, OLS_pytorch
from utils.helper import save_dict,load_dict, saveasnii


def get_activations(activations_dir, layer_name):
    """This function loads neural network features/activations (preprocessed using PCA) into a
    numpy array according to a given layer.

    Parameters
    ----------
    activations_dir : str
        Path to PCA processed Neural Network features
    layer_name : str
        which layer of the neural network to load,

    Returns
    -------
    train_activations : np.array
        matrix of dimensions # train_vids x # pca_components
        containing activations of train videos
    test_activations : np.array
        matrix of dimensions # test_vids x # pca_components
        containing activations of test videos
    """

    train_file = os.path.join(activations_dir,"train_" + layer_name + ".npy")
    test_file = os.path.join(activations_dir,"test_" + layer_name + ".npy")
    train_activations = np.load(train_file)
    test_activations = np.load(test_file)
    scaler = StandardScaler()  # normalization, z=(x-u)/s, u mean, s std
    train_activations = scaler.fit_transform(train_activations)  # 对模型输入进行归一化
    test_activations = scaler.fit_transform(test_activations)

    return train_activations, test_activations


def get_fmri(fmri_dir, ROI):
    """This function loads fMRI data into a numpy array for to a given ROI.

    Parameters
    ----------
    fmri_dir : str
        path to fMRI data.
    ROI : str
        name of ROI.

    Returns
    -------
    np.array
        matrix of dimensions #train_vids x #repetitions x #voxels
        containing fMRI responses to train videos of a given ROI
    """


    # Loading ROI data
    ROI_file = os.path.join(fmri_dir, ROI + ".pkl")
    ROI_data = load_dict(ROI_file)

    # averaging ROI data across repetitions
    ROI_data_train = np.mean(ROI_data["train"], axis = 1)  # 重复多次的值都取平均了
    if ROI == "WB":
        voxel_mask = ROI_data['voxel_mask']
        return ROI_data_train, voxel_mask

    return ROI_data_train


def predict_fmri_fast(train_activations, test_activations, train_fmri, fmri_test):
    """This function fits a linear regressor using train_activations and train_fmri,
    then returns the predicted fmri_pred_test using the fitted weights and
    test_activations.

    Parameters
    ----------
    train_activations : np.array
        matrix of dimensions #train_vids x #pca_components 
        containing activations of train videos.
    test_activations : np.array
        matrix of dimensions #test_vids x #pca_components
        containing activations of test videos
    train_fmri : np.array
        matrix of dimensions #train_vids x  #voxels
        containing fMRI responses to train videos
    use_gpu : bool
        Description of parameter `use_gpu`.

    Returns
    -------
    fmri_pred_test: np.array
        matrix of dimensions #test_vids x #voxels
        containing predicted fMRI responses to test videos .
    """
    # MultiOutputRegressor
    from sklearn.multioutput import MultiOutputRegressor
    model_ = MultiOutputRegressor(LinearRegression(), n_jobs=-1)
    model_.fit(train_activations, train_fmri) # [900, 100] [900, 368]
    fmri_pred_test = model_.predict(test_activations)  # 

    # import torch.nn as nn
    # from utils.mlp import NN_works, Mlp
    # mlp = Mlp(train_activations.shape[1], train_fmri.shape[1]).cuda()
    # loss_func = nn.MSELoss()
    # optimizer = torch.optim.AdamW(mlp.parameters(), 0.0003)
    # nn_work = NN_works(mlp, train_activations, train_fmri, test_activations, loss_func, optimizer, epoch=50)
    # temporary_result = nn_work.predict()
    # print("MSE before {}".format(mean_squared_error(fmri_test, temporary_result)))
    # mlp = nn_work.fit()
    # fmri_pred_test = nn_work.predict()  
    # print("MSE after  {}".format(mean_squared_error(fmri_test, fmri_pred_test)))

    # reg = OLS_pytorch(use_gpu)
    # # Step 2.学习feature至fMRI Response的映射关系θ
    # reg.fit(train_activations, train_fmri.T)  # 训练
    # # Step 3.使用θ来预测测试视频（处理成feature）的响应
    # fmri_pred_test = reg.predict(test_activations)  # 预测
    # import pdb
    # pdb.set_trace()

    return fmri_pred_test



def main():

    parser = argparse.ArgumentParser(description='Encoding model analysis for Algonauts 2021')
    parser.add_argument('-rd','--result_dir', help='saves predicted fMRI activity',default = './results', type=str)
    parser.add_argument('-ad','--activation_dir',help='directory containing DNN activations',default = './alexnet/', type=str)
    parser.add_argument('-model','--model',help='model name under which predicted fMRI activity will be saved', default = 'alexnet_mlp', type=str) # 存储
    parser.add_argument('-l','--layer',help='layer from which activations will be used to train and predict fMRI activity', default = 'layer_5', type=str)
    parser.add_argument('-sub','--sub',help='subject number from which real fMRI data will be used', default = 'sub04', type=str)
    parser.add_argument('-r','--roi',help='brain region, from which real fMRI data will be used', default = 'EBA', type=str)
    parser.add_argument('-m','--mode',help='test or val, val returns mean correlation by using 10% of training data for validation', default = 'val', type=str)
    parser.add_argument('-fd','--fmri_dir',help='directory containing fMRI activity', default = '/home/hc/Algonauts2021/data/participants_data_v2021', type=str)
    parser.add_argument('-v','--visualize',help='visualize whole brain results in MNI space or not, only available if -roi WB', default = True, type=bool)
    parser.add_argument('-b', '--batch_size',help='number of voxel to fit at one time in case of memory constraints', default=1000, type=int)
    args = vars(parser.parse_args())


    mode = args['mode'] # test or val
    sub = args['sub']
    ROI = args['roi']
    model = args['model']
    layer = args['layer']
    visualize_results = args['visualize']
    batch_size = args['batch_size'] # number of voxel to fit at one time in case of memory constraints

    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    if ROI == "WB": # whole brain
        track = "full_track"
    else:
        track = "mini_track"

    activation_dir = os.path.join(args['activation_dir'], 'pca_100')
    fmri_dir = os.path.join(args['fmri_dir'], track)

    sub_fmri_dir = os.path.join(fmri_dir, sub) # 指定受试者subXX
    results_dir = os.path.join(args['result_dir'], args['model'], args['layer'], track, sub)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("ROi is : ", ROI)

    train_activations,test_activations = get_activations(activation_dir, layer)
    if track == "full_track":
        fmri_train_all, voxel_mask = get_fmri(sub_fmri_dir, ROI)
    else:
        fmri_train_all = get_fmri(sub_fmri_dir, ROI)
    num_voxels = fmri_train_all.shape[1]


    if mode == 'val':
        # Here as an example we use first 900 videos as training and rest of the videos as validation
        test_activations = train_activations[900:,:]
        train_activations = train_activations[:900,:]
        fmri_train = fmri_train_all[:900,:]
        fmri_test = fmri_train_all[900:,:]  # train的后100例
        pred_fmri = np.zeros_like(fmri_test)
        pred_fmri_save_path = os.path.join(results_dir, ROI + '_val.npy')
    else:
        fmri_train = fmri_train_all
        num_test_videos = 102
        pred_fmri = np.zeros((num_test_videos, num_voxels))
        pred_fmri_save_path = os.path.join(results_dir, ROI + '_test.npy')


    print("number of voxels is ", num_voxels)
    iter = 0
    # 
    while iter < num_voxels-batch_size:  # batch_size means output_feature dimension
        pred_fmri[:,iter:iter+batch_size] = predict_fmri_fast(train_activations,test_activations,fmri_train[:,iter:iter+batch_size], fmri_test[:,iter:iter+batch_size])
        iter = iter+batch_size
        print((100*iter)//num_voxels," percent complete")
    
    # train_activations [900,100], test_activations [100,100], fmri_train[900, 368]
    # pred_fmri[:,iter:] = predict_fmri_fast(train_activations, test_activations, fmri_train[:,iter:iter+batch_size], use_gpu = use_gpu)
    pred_fmri[:,iter:] = predict_fmri_fast(train_activations, test_activations, fmri_train[:,iter:iter+batch_size], fmri_test[:,iter:iter+batch_size])

    if mode == 'val':
        need_log = False
        if need_log == True:
            log_dir = "./log/{}/{}/{}".format(model, layer, track)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)

            date_ = time.strftime("%a_%m%d_%H",time.localtime(time.time()))
            log_path = os.path.join(log_dir, date_)
            import logging
            logging.basicConfig(level=logging.INFO, format='[%(asctime)s] : %(message)s', 
                            datefmt = '%Y-%m-%d %a %H:%M:%S', 
                            handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
            logger = logging.getLogger('SLR')
            
            score = vectorized_correlation(fmri_test, pred_fmri) # 只有val mode 才会输出分数(相关系数ρ)
            mse = mean_squared_error(fmri_test, pred_fmri) # 

            logger.info("ROI: {}, Sub: {}, Mean correlation {}, MSE {}".format(ROI, sub, round(score.mean(),3), round(mse.mean(), 4)))
        else:
            score = vectorized_correlation(fmri_test, pred_fmri) # 只有val mode 才会输出分数(相关系数ρ)
            mse = mean_squared_error(fmri_test, pred_fmri) # 
            print("ROI: {}, Sub: {}, Mean correlation {}, MSE {}".format(ROI, sub, round(score.mean(),3), round(mse.mean(), 4)))
        # print("----------------------------------------------------------------------------")
        # print("Mean correlation for ROI : ",ROI, "in ",sub, " is :", round(score.mean(), 3))

        # result visualization for whole brain (full_track)
        if track == "full_track" and visualize_results:
            visual_mask_3D = np.zeros((78,93,71))
            visual_mask_3D[voxel_mask==1]= score
            brain_mask = './example.nii'
            nii_save_path =  os.path.join(results_dir, ROI + '_val.nii')
            saveasnii(brain_mask,nii_save_path, visual_mask_3D)
            view = plotting.view_img_on_surf(nii_save_path, threshold=None, surf_mesh='fsaverage',\
                                            title = 'Correlation for ' + sub, colorbar=True) # 需要Colorbar
            view_save_path = os.path.join(results_dir, ROI + '_val.html')  # 
            view.save_as_html(view_save_path)
            print("Results saved in this directory: ", results_dir)
            view.open_in_browser()

    np.save(pred_fmri_save_path, pred_fmri)


    print("----------------------------------------------------------------------------")
    print("ROI done : ", ROI)

if __name__ == "__main__":
    main()
    
    # y是多维度(n_samples, m_targets)情况下，使用MultiOutputRegressor, 如下
    # from sklearn.datasets import make_regression
    # from sklearn.multioutput import MultiOutputRegressor
    # from sklearn.linear_model import LinearRegression
    
    # X, y = make_regression(n_samples=50, n_features=100, n_targets=368)  # 
    # z = MultiOutputRegressor(SVR()).fit(X,y).predict(X)
    # print('ok')
    

# -*- coding: utf-8 -*-
import numpy as np
import os
import random
import argparse
import itertools
import nibabel as nib
from numpy.core.numeric import cross
from nilearn import plotting
from tqdm import tqdm

from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import torch
from torch.nn import MSELoss
from torch.utils.data import DataLoader
import time
import pickle
from tqdm import tqdm
from utils.ols import vectorized_correlation, OLS_pytorch
from utils.helper import save_dict, load_dict, saveasnii


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
    train_activations = scaler.fit_transform(train_activations)
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
    ROI_data_train = np.mean(ROI_data["train"], axis = 1)  # 重复多次的值都取平均, (1000, 3, x)
    if ROI == "WB":
        voxel_mask = ROI_data['voxel_mask']
        return ROI_data_train, voxel_mask

    return ROI_data_train


def predict_fmri_fast(train_activations, test_activations, train_fmri, use_gpu=False):
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
    # from sklearn.svm import SVR  # 
    model_lr = MultiOutputRegressor(LinearRegression(), n_jobs=-1) # 线性回归 LinearRegression

    num_cv = 5
    from sklearn.model_selection import cross_validate  # 交叉验证
    res = cross_validate(model_lr, X=train_activations, y=train_fmri, cv=num_cv, n_jobs=-1, return_estimator=True)
    
    test_pred_list = []
    for idx in range(num_cv):
        model_cv = res["estimator"][idx]  # 模型
        test_pred = model_cv.predict(test_activations)  # 在测试集上预测
        test_pred_list.append(test_pred)

    assert len(test_pred_list)== num_cv, "Here !!!!!!!!"
    fmri_pred_test = np.zeros_like(test_pred_list[0], dtype=np.float32)
    for elem in test_pred_list:  # 求和
        fmri_pred_test += elem
    fmri_pred_test /= num_cv  # 取平均

    # model_svr.fit(train_activations, train_fmri) # training [900, 100] [900, 368]
    # fmri_pred_test = model_svr.predict(test_activations)  # test

    # reg = OLS_pytorch(use_gpu)
    # # Step 2.学习feature至fMRI Response的映射关系θ
    # reg.fit(train_activations, train_fmri.T)  # 训练
    # # Step 3.使用θ来预测测试视频（处理成feature）的响应
    # fmri_pred_test = reg.predict(test_activations)  # 预测
    return fmri_pred_test


def predict_fmri_finetuned(video_list, fmri_nda, num_voxels):
    """
    使用网络训练，固定前面层不变（feature extraction），后面层作为voxel-encoding model部分训练
    """
    # Step0. split data, train-val-test
    print("Num voxels:", num_voxels)
    train_video, val_video, test_video = video_list[:900], video_list[900:1000], video_list[1000:] # 
    train_fmri, val_fmri = fmri_nda[:900], fmri_nda[900:]
    
    # Step1. Dataset and DataLoader
    from utils.custom_data import CustomDataset
    tra_ds = CustomDataset(train_video, train_fmri, mode="train")
    val_ds = CustomDataset(val_video, val_fmri, mode="val")

    tra_dl = DataLoader(tra_ds, batch_size=6, shuffle=True, num_workers=6)
    val_dl = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=4)
    
    test_ds = CustomDataset(test_video, None, mode="test")
    test_dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=2)
    

    # Step2. load pretrained model & modified networks
    from torch import hub
    hub.set_dir("/home/hc/Algonauts2021/Algo2021_hc/model_save")
    pretrained_model = hub.load('facebookresearch/semi-supervised-ImageNet1K-models', "resnext101_32x8d_swsl")
    from model_save.custom_resnext101_32x8d import Custom_ResNeXt101_32x8d as custom_net
    model = custom_net(pretrained_model, num_voxels).cuda()
    
    # Step3. Training and Testing.
    #######################
    # Training Processing #
    #######################
    num_epoch = 1000
    loss_func = MSELoss()
    from torch.optim import Adam
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=100, gamma=0.2)

    for e in range(num_epoch):
        model.train()
        epoch_loss = 0
        step = 0
        for a,b in tra_dl:
            step += 1
            input = a.to(torch.float32).cuda()
            label = b.to(torch.float32).cuda()
            optimizer.zero_grad()
            output = model(input)
            loss = loss_func(output, label)
            loss.backward()
            epoch_loss += loss.item()
            optimizer.step()
            if (step+1) % 30 == 0:
                print("Epoch {:03}/{:03} | Step {:03} | Loss: {:.4}".format(e+1, num_epoch, step+1, loss.item()))
        print("Mean Loss: {:.4}".format(epoch_loss))

        # val
        if (e+1) % 1 == 0:
            with torch.no_grad():
                model.eval()
                val_score = 0
                step_ = 0
                for c,d in val_dl:
                    step += 1
                    input = c.cuda()
                    label = d
                    pred = model(input)
                    score = vectorized_correlation(pred.cpu().numpy(), label.numpy())
                    val_score += score.mean()
                print("Validation Score {:.5f}".format(val_score/step))

    import pdb
    pdb.set_trace()


    ##############
    # Prediction #
    ##############

    return 1
    # return fmri_pred_test



def encoding(result_dir='./results', activation_dir='./activations/resnet/...', 
            model='encoding_test', layer='layer_3', sub='sub04', ROI='EBA', mode='test', 
            fmri_dir='/home/hc/Algonauts2021/data/participants_data_v2021', visualize_results=False,
            batch_size=1000, n_pca=100):

    # parser = argparse.ArgumentParser(description='Encoding model analysis for Algonauts 2021')
    # parser.add_argument('-rd','--result_dir', help='saves predicted fMRI activity',default = './results', type=str)
    # parser.add_argument('-ad','--activation_dir',help='directory containing DNN activations',default = './alexnet/', type=str)
    # parser.add_argument('-model','--model',help='model name under which predicted fMRI activity will be saved', default = 'alexnet_devkit', type=str)
    # parser.add_argument('-l','--layer',help='layer from which activations will be used to train and predict fMRI activity', default = 'layer_5', type=str)
    # parser.add_argument('-sub','--sub',help='subject number from which real fMRI data will be used', default = 'sub04', type=str)
    # parser.add_argument('-r','--roi',help='brain region, from which real fMRI data will be used', default = 'EBA', type=str)
    # parser.add_argument('-m','--mode',help='test or val, val returns mean correlation by using 10% of training data for validation', default = 'val', type=str)
    # parser.add_argument('-fd','--fmri_dir',help='directory containing fMRI activity', default = '/openbayes/input/input0/participants_data_v2021', type=str)
    # parser.add_argument('-v','--visualize',help='visualize whole brain results in MNI space or not, only available if -roi WB', default = True, type=bool)
    # parser.add_argument('-b', '--batch_size',help=' number of voxel to fit at one time in case of memory constraints', default = 1000, type=int)
    # parser.add_argument('-pca', '--pca_components', help='n_components of PCA', default=100, type=int)
    # args = vars(parser.parse_args())

    # result_dir = args['result_dir']
    # activation_dir = args['activation_dir']
    # model = args['model']
    # layer = args['layer']
    # sub = args['sub']
    # ROI = args['roi']
    # mode = args['mode'] # test or val
    # fmri_dir = args['fmri_dir']
    # visualize_results = args['visualize']
    # batch_size = args['batch_size'] # number of voxel to fit at one time in case of memory constraints
    # n_pca = args['pca_components']
    
    if torch.cuda.is_available():
        use_gpu = True
    else:
        use_gpu = False

    if ROI == "WB": # whole brain
        track = "full_track"
    else:
        track = "mini_track"

    activation_dir = os.path.join(activation_dir, f'pca_{n_pca}')
    fmri_dir = os.path.join(fmri_dir, track)

    sub_fmri_dir = os.path.join(fmri_dir, sub)
    results_dir = os.path.join(result_dir, model, layer, track, sub)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    print("ROi is : ", ROI)

    ##############################
    # 
    ##############################

    # 1.获取video_list（input）
    from glob import glob
    video_dir = r"/home/hc/Algonauts2021/data/AlgonautsVideos268_All_30fpsmax"
    video_list = glob(video_dir + '/*.mp4')
    video_list.sort()
    
    # 2.读取fmri_nda（label）
    if track == "full_track":
        fmri_train_all, voxel_mask = get_fmri(sub_fmri_dir, ROI)
    else:
        fmri_train_all = get_fmri(sub_fmri_dir, ROI)
    num_voxels = fmri_train_all.shape[1]
    
    # # 3.train set, val set, test set
    # train_video, val_video, test_video = video_list[:900], video_list[900:1000], video_list[1000:]
    # train_fmri, val_fmri = fmri_train_all[:900], fmri_train_all[900:1000]

    # 4. Dataset and Dataloader
    
    fmri_pred_test = predict_fmri_finetuned(video_list, fmri_train_all, num_voxels)
    
    # # train_activations,test_activations = get_activations(activation_dir, layer)  # 训练，测试的输入
    import pdb
    pdb.set_trace()


    # # if mode == 'val':
    # #     # Here as an example we use first 900 videos as training and rest of the videos as validation
    # #     test_activations = train_activations[900:,:]
    # #     train_activations = train_activations[:900,:]
    # #     fmri_train = fmri_train_all[:900,:]
    # #     fmri_test = fmri_train_all[900:,:]  # train的后100例
    # #     pred_fmri = np.zeros_like(fmri_test)
    # #     pred_fmri_save_path = os.path.join(results_dir, ROI + '_val.npy')
    # # else:
    # #     fmri_train = fmri_train_all
    # #     num_test_videos = 102
    # #     pred_fmri = np.zeros((num_test_videos, num_voxels))
    # #     pred_fmri_save_path = os.path.join(results_dir, ROI + '_test.npy')

    # # print("number of voxels is ", num_voxels)
    # # iter = 0
    # # while iter < num_voxels-batch_size:
    # #     pred_fmri[:,iter:iter+batch_size] = predict_fmri_finetuned(train_activations,test_activations,fmri_train[:,iter:iter+batch_size], use_gpu = use_gpu)
    # #     iter = iter+batch_size
    # #     print((100*iter)//num_voxels," percent complete")
    
    # # # train_activations [900,100], test_activations [100,100], fmri_train[900, 368]
    # # pred_fmri[:,iter:] = predict_fmri_finetuned(train_activations, test_activations, fmri_train[:,iter:iter+batch_size], use_gpu = use_gpu)

    # if mode == 'val':
    #     log_dir = "./log/{}/{}/{}".format(model, layer, track)
    #     if not os.path.exists(log_dir):
    #         os.makedirs(log_dir)

    #     date_ = time.strftime("%a_%m%d_%H",time.localtime(time.time()))
    #     log_path = os.path.join(log_dir, date_)
    #     import logging
    #     logging.basicConfig(level=logging.INFO, format='[%(asctime)s] : %(message)s', 
    #                     datefmt = '%Y-%m-%d %a %H:%M:%S', 
    #                     handlers=[logging.FileHandler(log_path), logging.StreamHandler()])
    #     logger = logging.getLogger('SLR')
        
    #     score = vectorized_correlation(fmri_test, pred_fmri) # 只有val mode 才会输出分数(相关系数ρ)
    #     mse = mean_squared_error(fmri_test, pred_fmri) # 

    #     logger.info("ROI: {}, Sub: {}, Mean correlation {}, MSE {}".format(ROI, sub, round(score.mean(),3), round(mse.mean(), 4)))
    #     # score = vectorized_correlation(fmri_test, pred_fmri) # only val mode, print(score)
    #     # print("----------------------------------------------------------------------------")
    #     # print("Mean correlation for ROI : ",ROI, "in ",sub, " is :", round(score.mean(), 3))

    #     # result visualization for whole brain (full_track)
    #     if track == "full_track" and visualize_results:
    #         visual_mask_3D = np.zeros((78,93,71))
    #         visual_mask_3D[voxel_mask==1]= score
    #         brain_mask = './example.nii'
    #         nii_save_path =  os.path.join(results_dir, ROI + '_val.nii')
    #         saveasnii(brain_mask,nii_save_path, visual_mask_3D)
    #         view = plotting.view_img_on_surf(nii_save_path, threshold=None, surf_mesh='fsaverage',\
    #                                         title = 'Correlation for ' + sub, colorbar=True) # 需要Colorbar
    #         view_save_path = os.path.join(results_dir, ROI + '_val.html')  # 
    #         view.save_as_html(view_save_path)
    #         print("Results saved in this directory: ", results_dir)
    #         view.open_in_browser()
    
    # np.save(pred_fmri_save_path, pred_fmri)
    # print("----------------------------------------------------------------------------")
    # print("ROI done : ", ROI)

    # if mode == 'val':
    #     return round(score.mean(),3), round(mse.mean(), 4)


if __name__ == "__main__":
   encoding(activation_dir='./activations/resnet/swsl_resnext101_32x8d')


###
# This file will:
# 1. Generate and save Alexnet features in a given folder
# 2. preprocess Alexnet features using PCA and save them in another folder
###
import glob
from vgg import *
import numpy as np
import urllib
import torch
import cv2
import argparse
import time
import random
import difflib
from tqdm import tqdm
from torchvision import transforms as trn
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.autograd import Variable as V
from sklearn.decomposition import PCA, IncrementalPCA

seed = 42
# Torch RNG
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
# Python RNG
np.random.seed(seed)
random.seed(seed)



def get_video_from_mp4(file, sampling_rate):
    """This function takes a mp4 video file as input and returns
    an array of frames in numpy format.

    Parameters
    ----------
    file : str
        path to mp4 video file
    sampling_rate : int
        how many frames to skip when doing frame sampling.

    Returns
    -------
    video: np.array

    num_frames: int
        number of frames extracted

    """
    cap = cv2.VideoCapture(file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((int(frameCount / sampling_rate), frameHeight,
                   frameWidth, 3), np.dtype('uint8'))
    fc = 0 # 
    ret = True
    while fc < frameCount and ret:
        fc += 1
        if fc % sampling_rate == 0:
            (ret, buf[int((fc - 1) / sampling_rate)]) = cap.read()

    cap.release()

    return np.expand_dims(buf, axis=0),int(frameCount / sampling_rate)


def load_model(model_checkpoints):
    """This function initializes an Alexnet and load
    its weights from a pretrained model
    ----------
    model_checkpoints : str
        model checkpoints location.

    Returns
    -------
    model
        pytorch model of alexnet

    """

    # from torchvision.models import resnet50, resnext101_32x8d
    from torch import hub
    hub.set_dir("/openbayes/home/Algonauts2021_devkit/model_save")
    model = hub.load('facebookresearch/semi-supervised-ImageNet1K-models', "resnet50_swsl")
    # model = resnext101_32x8d()
    model_file = model_checkpoints
    checkpoint = torch.load(model_file)
    model.load_state_dict(checkpoint)
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    return model


def get_activations_and_save(model, video_list, activations_dir, sampling_rate = 4):
    """This function generates Alexnet features and save them in a specified directory.

    Parameters
    ----------
    model :
        pytorch model : vgg.
    video_list : list
        the list contains path to all videos.
    activations_dir : str
        save path for extracted features.
    sampling_rate : int
        how many frames to skip when feeding into the network.

    """
    # 一些预处理
    centre_crop = trn.Compose([
            trn.ToPILImage(),
            trn.Resize((224,224)),
            trn.ToTensor(),
            trn.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def get_activations(module, fea_in, fea_out): # hooker
        activations_per_frame.append(fea_out)  # 获取特征
        
    for idx, child in enumerate(model.children()):  # 注册hooker
        if idx>=4 and idx<=7:  # 指定层
            child.register_forward_hook(hook=get_activations)

    dir_file = os.listdir(activations_dir)

    for video_file in tqdm(video_list):
        video_file_name = os.path.split(video_file)[-1].split(".")[0] # basename without suffix
        res = difflib.get_close_matches(video_file_name, dir_file, 3, cutoff=0.6)  # 模糊匹配, 判断是否已经提取过特征
        if len(res) != 0:  # 若有文件，则跳过
            break
        vid, num_frames = get_video_from_mp4(video_file, sampling_rate) # vid [1, 22, 268, 268, 3]
        activations = [] # list

        for frame in range(num_frames): # 采样得到的每一帧
            activations_per_frame = []
            img =  vid[0,frame,:,:,:]
            input_img = V(centre_crop(img).unsqueeze(0)) # 预处理，V=Variable
            if torch.cuda.is_available():
                input_img=input_img.cuda()
            x = model.forward(input_img) # 每帧都送进模型
            assert len(activations_per_frame) == 4, f'ERROR!, len is {len(activations_per_frame)}'
            for i, feat in enumerate(activations_per_frame): # 得到不同层的特征
                if frame==0:
                    activations.append(feat.data.cpu().numpy().ravel()) # ravel -> flatten
                else:
                    activations[i] = activations[i] + feat.data.cpu().numpy().ravel() # 每一帧的累加
        for layer in range(len(activations)):
            save_path = os.path.join(activations_dir, video_file_name+"_"+"layer" + "_" + str(layer+1) + ".npy")
            np.save(save_path, activations[layer]/float(num_frames))  # 此处将每帧累加的激活值除以帧数，是取平均


def do_PCA_and_save(activations_dir, save_dir, n_components):
    """This function preprocesses Neural Network features using PCA and save the results
    in  a specified directory

    Parameters
    ----------
    activations_dir : str
        save path for extracted features.
    save_dir : str
        save path for extracted PCA features.

    """

    layers = ['layer_1','layer_2','layer_3','layer_4']  # 'layer_5','layer_6','layer_7','layer_8'
    n_components = n_components
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for layer in tqdm(layers):
        activations_file_list = glob.glob(activations_dir +'/*'+layer+'.npy')
        activations_file_list.sort()
        feature_dim = np.load(activations_file_list[0])
        x = np.zeros((len(activations_file_list),feature_dim.shape[0]))
        for i, activation_file in enumerate(activations_file_list):
            temp = np.load(activation_file)
            x[i,:] = temp
        x_train = x[:1000,:]  # 前1k例作为训练集
        x_test = x[1000:,:]  # 1k之后(102例)得到作为测试集

        start_time = time.time()
        x_test = StandardScaler().fit_transform(x_test)
        x_train = StandardScaler().fit_transform(x_train)
        ipca = PCA(n_components=n_components) #, batch_size=20)
        ipca.fit(x_train)

        x_train = ipca.transform(x_train)
        x_test = ipca.transform(x_test)
        train_save_path = os.path.join(save_dir,"train_"+layer)
        test_save_path = os.path.join(save_dir,"test_"+layer)
        np.save(train_save_path,x_train)
        np.save(test_save_path,x_test)

def main():
    parser = argparse.ArgumentParser(description='Feature Extraction from Alexnet and preprocessing using PCA')
    parser.add_argument('-vdir','--video_data_dir', help='video data directory',default = '/openbayes/input/input0/AlgonautsVideos268_All_30fpsmax/', type=str)
    parser.add_argument('-sdir','--save_dir', help='saves processed features',default = './features_from_models/resnet', type=str)
    args = vars(parser.parse_args())

    save_dir=args['save_dir']  # net中层特征的 存储路径
    if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    video_dir = args['video_data_dir']  # 视频数据存放路径
    video_list = glob.glob(video_dir + '/*.mp4')
    video_list.sort()
    print('Total Number of Videos: ', len(video_list))

    # load Alexnet
    # Download pretrained Alexnet from:
    # https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth
    # and save in the current directory
    checkpoint_path = "./model_save/checkpoints/resnet50-19c8e357.pth"  # 预训练模型下载
    # if not os.path.exists(checkpoint_path):
    #     url = "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth"
    #     urllib.request.urlretrieve(url, checkpoint_path)
    model = load_model(checkpoint_path)

    # get and save activations
    activations_dir = os.path.join(save_dir)
    if not os.path.exists(activations_dir):
        os.makedirs(activations_dir)
    print("-------------Saving activations ----------------------------")
    get_activations_and_save(model, video_list, activations_dir) # 采样率4

    # preprocessing using PCA and save
    n_components = 100
    pca_dir = os.path.join(save_dir, f'pca_{n_components}')
    print("-------------performing  PCA----------------------------")
    do_PCA_and_save(activations_dir, pca_dir, n_components)


if __name__ == "__main__":
    main()

